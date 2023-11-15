import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_points3d.applications.pointnet2 import PointNet2
import numpy as np
import scipy
from torch_geometric.data import Data
import irispy


class CorridorGenerationNetworkLSTM(nn.Module):
    def __init__(self, pcl_input_nc, pcl_output_nc, state_input_size, state_output_size, output_channels, hidden_size,
                 lr=1e-3, T_0=500, T_mult=1, eta_min=1e-5, last_epoch=-1, max_elements=500, use_scheduler=True):
        super(CorridorGenerationNetworkLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.state_output_size = state_output_size

        self.pcl_encoder = PointNet2("encoder", input_nc=pcl_input_nc, output_nc=pcl_output_nc,
                                     num_layers=3, multiscale=True)
        self.pcl_encoder.to(self.device)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_size, state_output_size)
        )
        self.state_encoder.to(self.device)

        self.lstm_num_layers = 1
        self.decoder = nn.LSTM(input_size=pcl_output_nc + state_output_size, hidden_size=hidden_size, num_layers=self.lstm_num_layers,
                               batch_first=True)
        self.decoder.to(self.device)

        # Get next element
        self.fc_element = nn.Linear(hidden_size, output_channels)
        self.fc_element.to(self.device)

        # Get the probability of reaching the end of the current polytope (last element)
        self.fc_elements_end_prob = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.fc_elements_end_prob.to(self.device)

        # Get the probability of reaching the end of the sequence (last polytope)
        self.fc_polytopes_end_prob = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.fc_polytopes_end_prob.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=T_0,
                                                                            T_mult=T_mult,
                                                                            eta_min=eta_min,
                                                                            last_epoch=last_epoch)

        self.max_elements = max_elements

    def prepare_pointnet_data(self, points):
        # Check the type of the input
        if isinstance(points, np.ndarray):
            # Copy the points to create features
            features = np.copy(points)

            # Convert to torch tensors
            points = torch.tensor(points, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.float32)

        elif isinstance(points, torch.Tensor):
            # Copy the points to create features
            features = points.clone()

            # Convert to float tensors
            points = points.float()
            features = features.float()

        else:
            raise TypeError("Input should be of type numpy.ndarray or torch.Tensor")

        features.to(self.device)
        points.to(self.device)
        # Create the Data object
        data = Data(x=features, pos=points)

        return data

    def forward(self, pcl, state):
        pcl_data = self.prepare_pointnet_data(pcl)

        pcl_features_data = self.pcl_encoder(pcl_data)

        pcl_features = pcl_features_data.x.squeeze(2)

        # Ensure state is a float tensor
        if state.dtype != torch.float32:
            state = state.float()

        state_features = self.state_encoder(state)
    
        # print("pcl_features dim: ", pcl_features.size())
        #print("state_features dim[1]: ", state_features.size()[1])

        # The size of output_features is equal to the size of state_features
        
        

        combined_features = torch.cat((pcl_features, state_features), dim=1)
        # print("combined_features dim: ", combined_features.size())
        batch_size = combined_features.size(0)
        combined_features = combined_features.unsqueeze(1)

        # output_embedding = torch.zeros(batch_size, 1, self.hidden_size).to(self.device)
        # print("output_embedding dim: ", output_embedding.size())
        
        h_0 = torch.zeros(self.lstm_num_layers, batch_size, self.decoder.hidden_size).to(self.device)
        c_0 = torch.zeros(self.lstm_num_layers, batch_size, self.decoder.hidden_size).to(self.device)

        predicted_elements = []
        elements_end_probs = []
        polytopes_end_probs = []

        # input_element = torch.zeros(batch_size, 1, combined_features.size(-1)).to(self.device)
        # input_element = torch.cat((combined_features, output_embedding), dim=-1)
        # print("input_element: ", input_element.size())

        hidden = (h_0, c_0)

        for _ in range(self.max_elements):
            output, hidden = self.decoder(combined_features, hidden)

            output_element = self.fc_element(output)
            curr_element_end_prob = self.fc_elements_end_prob(output)
            curr_polytope_end_prob = self.fc_polytopes_end_prob(output)

            predicted_elements.append(output_element)
            elements_end_probs.append(curr_element_end_prob)
            polytopes_end_probs.append(curr_polytope_end_prob)

            # print("output: ", output.size())

            # input_element = torch.cat((combined_features, output), dim=-1)
            # print("input_element: ", input_element.size())

        predicted_elements = torch.cat(predicted_elements, dim=1)
        element_end_probs = torch.cat(elements_end_probs, dim=1)
        polytope_end_probs = torch.cat(polytopes_end_probs, dim=1)

        return predicted_elements, element_end_probs, polytope_end_probs

    def reconstruct_corridor_sequence(self, predicted_elements, element_end_probs, polytope_end_probs, binary_threshold=0.5, type="hpoly"):
        polyhedrons = []

        cur_elems = None
        if type == "hpoly":
            # print("polytope_end_probs: ", polytope_end_probs)
            for i in range(len(polytope_end_probs)):
                # print("polytope_end_probs[0]: ", polytope_end_probs[0])
                if polytope_end_probs[0][i][0] <= binary_threshold:
                    if element_end_probs[0][i][0] < binary_threshold:
                        if cur_elems is None:
                            cur_elems = predicted_elements[i]
                        else:
                            cur_elems = torch.cat((cur_elems, predicted_elements[i]), dim=0)
                    else:
                        if cur_elems is not None:
                            cur_poly = irispy.Polyhedron()
                            cur_poly.setA(cur_elems[:, :-1].detach().cpu().numpy())
                            cur_poly.setB(cur_elems[:, -1].detach().cpu().numpy())
                            polyhedrons.append(cur_poly)
                            cur_elems = None

                if polytope_end_probs[0][i][0] > binary_threshold:
                    break

        elif type == "vpoly":
            raise NotImplementedError("vpoly to corridor sequence has not implemented yet")
        else:
            raise ValueError("Unknown type of polytope")

        return polyhedrons

    def corridor_sequence_inference(self, pcl, state):
        pred_elements, pred_elements_end_prob, pred_polytopes_end_prob = self.forward(pcl, state)
        polyhedrons = self.reconstruct_corridor_sequence(pred_elements, pred_elements_end_prob, pred_polytopes_end_prob)
        return polyhedrons

    def get_inner_pts(self, state, polyhedrons, eps=0.01):
        start = state[:3]
        goal = state[3:6]
        n = len(polyhedrons)

        if n <= 0:
            print("No corridor no path!")
            return None

        if n == 1:
            pt = 0.5 * (goal + start)
            return np.array([pt])

        inner_pts = []
        for i in range(n - 1):
            poly1 = polyhedrons[i]
            poly2 = polyhedrons[i + 1]

            poly1_hpoly = np.hstack((poly1.getA(), poly1.getB().reshape(-1, 1)))
            poly2_hpoly = np.hstack((poly2.getA(), poly2.getB().reshape(-1, 1)))

            total_constraints = np.vstack((poly1_hpoly, poly2_hpoly))

            A = total_constraints[:, 0:3]
            b = total_constraints[:, 3]
            c = [0, 0, 0, -1]
            A = np.hstack((A, np.ones(len(b)).reshape(len(b), 1)))

            bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
            minmaxsd = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)

            if minmaxsd.fun is None:
                return None
            inner_pts.append(minmaxsd.x[0:3])

        return np.asarray(inner_pts)

    def loss_function(self, pred_elements, pred_elements_end_prob, pred_polytopes_end_prob,
                      gt_elements, gt_elements_end_prob, gt_polytopes_end_prob,
                      element_loss_weight=1.0, elements_end_prob_loss_weight=1.0, polytopes_end_prob_loss_weight=1.0):
        
        gt_elements_end_prob = gt_elements_end_prob.float().unsqueeze(-1)
        gt_polytopes_end_prob = gt_polytopes_end_prob.float().unsqueeze(-1)

        pos_weight_elements = (gt_elements_end_prob == 0).sum().float() / (gt_elements_end_prob == 1).sum().float()
        pos_weight_polytopes = (gt_polytopes_end_prob == 0).sum().float() / (gt_polytopes_end_prob == 1).sum().float()

        element_loss = torch.nn.MSELoss()(pred_elements, gt_elements)
        elements_end_prob_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_elements]).to(self.device))(pred_elements_end_prob, gt_elements_end_prob)
        polytopes_end_prob_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_polytopes]).to(self.device))(pred_polytopes_end_prob, gt_polytopes_end_prob)

        total_loss = element_loss_weight * element_loss + elements_end_prob_loss_weight * elements_end_prob_loss + polytopes_end_prob_loss_weight * polytopes_end_prob_loss

        return total_loss, element_loss, elements_end_prob_loss, polytopes_end_prob_loss

    def mask_padding_for_loss(self, pred_elements, pred_elements_end_prob, pred_polytopes_end_prob, gt_elements, gt_polytopes_end_prob):
        end_idices = torch.where(gt_polytopes_end_prob == 1)
        for i, end_idx in enumerate(end_idices[1]):
            # print("end_idx: ", end_idx)
            mask_2d = torch.arange(gt_elements.size(1)).to(self.device) <= end_idx
            mask_2d = mask_2d.unsqueeze(-1)

            pred_elements[i] = pred_elements[i] * mask_2d
            pred_elements_end_prob[i] = pred_elements_end_prob[i] * mask_2d
            pred_polytopes_end_prob[i] = pred_polytopes_end_prob[i] * mask_2d

        return pred_elements, pred_elements_end_prob, pred_polytopes_end_prob

    def train_model(self, pcl, state, gt_elements, gt_elements_end_prob, gt_polytopes_end_prob):
        self.optimizer.zero_grad()

        pred_elements, pred_elements_end_prob, pred_polytopes_end_prob = self.forward(pcl, state)
        masked_pred_elements, masked_pred_elements_end_prob, masked_pred_polytopes_end_prob = self.mask_padding_for_loss(pred_elements, pred_elements_end_prob, pred_polytopes_end_prob, gt_elements, gt_polytopes_end_prob)
        
        gt_elements = gt_elements.float()
        gt_elements_end_prob = gt_elements_end_prob.float()
        gt_polytopes_end_prob = gt_polytopes_end_prob.float()
        masked_pred_elements = masked_pred_elements.float()
        masked_pred_elements_end_prob = masked_pred_elements_end_prob.float()
        masked_pred_polytopes_end_prob = masked_pred_polytopes_end_prob.float()
        
        total_loss, element_loss, elements_end_prob_loss, polytopes_end_prob_loss = self.loss_function(masked_pred_elements,
                                                                                                       masked_pred_elements_end_prob,
                                                                                                       masked_pred_polytopes_end_prob,
                                                                                                       gt_elements,
                                                                                                       gt_elements_end_prob,
                                                                                                       gt_polytopes_end_prob)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.optimizer.step()

        return total_loss.item(), element_loss.item(), elements_end_prob_loss.item(), polytopes_end_prob_loss.item()

    def eval_model(self, pcl, state, gt_elements, gt_elements_end_prob,
                   gt_polytopes_end_prob):
        pred_elements, pred_elements_end_prob_tensor, pred_polytopes_end_prob_tensor = self.forward(pcl, state)
        total_loss, element_loss, elements_end_prob_loss, polytopes_end_prob_loss = self.loss_function(pred_elements,
                                                                                                       pred_elements_end_prob_tensor,
                                                                                                       pred_polytopes_end_prob_tensor,
                                                                                                       gt_elements,
                                                                                                       gt_elements_end_prob,
                                                                                                       gt_polytopes_end_prob)

        pred_elements_end_prob = pred_elements_end_prob_tensor.tolist()
        pred_polytopes_end_prob = pred_polytopes_end_prob_tensor.tolist()

        polyhedrons = self.reconstruct_corridor_sequence(pred_elements, pred_elements_end_prob, pred_polytopes_end_prob)
        inner_pts = self.get_inner_pts(state, polyhedrons)

        if inner_pts is None:
            corridor_continuity = False
        else:
            corridor_continuity = True

        # Calculate cosine similarities
        # print("gt_elements dim: ", gt_elements.size())
        # print("pred_elements dim: ", pred_elements.size())
        # print("gt_elem_end_prob dim: ", gt_elements_end_prob.size())
        # print("pred_elem_end_prob dim: ", pred_elements_end_prob_tensor.size())
        # print("gt_seq_end_prob dim: ", gt_polytopes_end_prob.size())
        # print("pred_seq_end_prob dim: ", pred_polytopes_end_prob_tensor.size())
        elem_cosine_similarity = F.cosine_similarity(gt_elements, pred_elements, dim=2)
        elem_cosine_similarity_mean = torch.mean(elem_cosine_similarity)
        # print("elem_cosine_similarity dim: ", elem_cosine_similarity.size())
        # print("elem_cosine_similarity: ", elem_cosine_similarity)
        
        gt_elements_end_prob = gt_elements_end_prob.float().unsqueeze(-1)
        elem_end_prob_cosine_similarity = F.cosine_similarity(gt_elements_end_prob, pred_elements_end_prob_tensor, dim=2)
        elem_end_prob_cosine_similarity_mean = torch.mean(elem_end_prob_cosine_similarity)
        # print("elem_end_prob_cosine_similarity dim: ", elem_end_prob_cosine_similarity.size())
        # print("elem_end_prob_cosine_similarity: ", elem_end_prob_cosine_similarity)

        gt_polytopes_end_prob = gt_polytopes_end_prob.float().unsqueeze(-1)
        seq_end_prob_cosine_similarity = F.cosine_similarity(gt_polytopes_end_prob, pred_polytopes_end_prob_tensor, dim=2)
        seq_end_prob_cosine_similarity_mean = torch.mean(seq_end_prob_cosine_similarity)
        # print("seq_end_prob_cosine_similarity dim: ", seq_end_prob_cosine_similarity.size())
        # print("seq_end_prob_cosine_similarity: ", seq_end_prob_cosine_similarity)

        return total_loss.item(), element_loss.item(), elements_end_prob_loss.item(), polytopes_end_prob_loss.item(), corridor_continuity, elem_cosine_similarity_mean.item(), elem_end_prob_cosine_similarity_mean.item(), seq_end_prob_cosine_similarity_mean.item()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
