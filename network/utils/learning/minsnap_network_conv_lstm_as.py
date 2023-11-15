import torch
import torch.nn as nn
import cvxpy as cp
from utils.learning.layers import OsqpLayer
from utils.min_traj_opt import MinTrajOpt
from functools import reduce
import operator
import numpy as np
import torch
#from cvxpylayers.torch import CvxpyLayer
import torch.optim as optim
import resource
import gc
from memory_profiler import profile
import objgraph
import tracemalloc
import copy
import sys
import time
from torch.autograd import Function, Variable




class ConvLSTMMinimalSnapNetwork(nn.Module):
    def __init__(self, seg, max_poly_length, lr=1e-3,
                 T_0=500, T_mult=1, eta_min=1e-5, last_epoch=-1, hidden_size=128, use_scheduler=True,
                 loss_dim_reduction='mean', w1=1, wt= 5.0, wc=1.0, wp=1.0):
        super().__init__()
        #init

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg = seg

        self.output_module = nn.LSTM(input_size=38, hidden_size=hidden_size, num_layers=1)
        self.tfs_output_layer = nn.Linear(hidden_size, 1)
        self.stop_token_output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(self.tfs_output_layer.weight, nonlinearity='relu')

        self.hidden_size = hidden_size

        self.output_module.to(self.device)
        self.tfs_output_layer.to(self.device)
        self.stop_token_output_layer.to(self.device)

        self.max_poly_length = max_poly_length


 
        self.hpoly_input_module = nn.Sequential(
            # yuwei: revise the size here
            nn.Conv2d(50, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16, 32)
        )
        self.state_input_module = nn.Sequential(
            nn.Conv1d(9, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8, 6)
        )

        self.hpoly_input_module.to(self.device)
        self.state_input_module.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=T_0,
                                                                            T_mult=T_mult,
                                                                            eta_min=eta_min,
                                                                            last_epoch=last_epoch)

        self.loss_dim_reduction = loss_dim_reduction

        self.w1 = w1
        self.wt = wt
        self.wc = wc
        self.wp = wp

        self.to(self.device)

        self.phase = 2


    def forward(self, stacked_state, stacked_hpolys):

        inference_time = 0.0

        osqplayer = OsqpLayer()
        qp_traj = MinTrajOpt([])

        inference_start_time = time.time()

        stacked_state = stacked_state.float()
        stacked_hpolys = stacked_hpolys.float()

        embedding_start_time = time.time()
        state_embeddings = self.state_input_module(stacked_state)
        hpoly_embeddings = self.hpoly_input_module(stacked_hpolys)
        embedding_end_time = time.time()
        inference_time += embedding_end_time - embedding_start_time

        state_embeddings = state_embeddings.squeeze(-1)
        hpoly_embeddings = hpoly_embeddings.squeeze(-1)

        combined = torch.cat((state_embeddings, hpoly_embeddings), dim=1).type(torch.float32)



        """
        tfs = self.output_module(combined)

        

        
        """

        tf = torch.empty((0,)).to(self.device)
        stop_tokens = torch.empty((0,)).to(self.device)

        h = torch.zeros(1, self.hidden_size).to(self.device)
        c = torch.zeros(1, self.hidden_size).to(self.device)

        for k in range(5):
            lstm_start_time = time.time()
            lstm_out, (h, c) = self.output_module(combined, (h, c))

            curr_tf_elem = self.tfs_output_layer(lstm_out)
            curr_stop_tokens = self.stop_token_output_layer(lstm_out)

            lstm_end_time = time.time()

            inference_time += lstm_end_time - lstm_start_time

            tf = torch.cat((tf, curr_tf_elem), dim=0)
            stop_tokens = torch.cat((stop_tokens, curr_stop_tokens), dim=0)


        stop_tokens = stop_tokens.permute(1, 0)
        tf = tf.permute(1, 0)

        if stop_tokens.shape[1] < 5:
            # print("shape of stop_tokens is : ", stop_tokens.shape)
            # print("shape of torch.ones(5 - stop_tokens.shape[0]).to(self.device) is : ",  torch.ones(5 - stop_tokens.shape[0]).to(self.device).shape)
            # Pad with 1
            stop_tokens_padding = torch.ones(5 - stop_tokens.shape[1])
            stop_tokens_padding = stop_tokens_padding.unsqueeze(0)
            stop_tokens = torch.cat((stop_tokens, stop_tokens_padding.to(self.device)), dim=1)
            # print("shape of stop_tokens is (<5) : ", stop_tokens.shape)
        # else:
        # print("shape of stop_tokens is (>=5) : ", stop_tokens.shape)

        # Check if tf has 5 elements
        if tf.shape[1] < 5:
            # Pad with 0
            tf_padding = torch.zeros(5 - tf.shape[1])
            tf_padding = tf_padding.unsqueeze(0)
            tf = torch.cat((tf, tf_padding.to(self.device)), dim=1)

        inference_end_time = time.time()

        print("===== Current tf from checkpoint is : ", tf)
        print("Current stop_tokens is : ", stop_tokens)

        print("Inference time is : ", inference_time)

        tf = tf[0].cpu()
        x_i = stacked_state[0].cpu()
        hpoly_i = stacked_hpolys[0].cpu()
        traj_times_i = None

        stop_tokens_i = stop_tokens[0].cpu()

        qp_traj.update(x_i, hpoly_i, tf, phase=self.phase, traj_times=traj_times_i)

        solution, curr_obj1_val, curr_objt_val, curr_objc_val, curr_stop_token_loss = osqplayer.forward4lstm(qp_traj, stop_tokens_i)

        return solution, curr_obj1_val, curr_objt_val, curr_objc_val, curr_stop_token_loss

    # x includes start-end state and hpolys
    # @profile
    def forward_batch(self, stacked_state, stacked_hpolys, traj_times):
        # objt outlier debug
        objt_debug_log = {"hpolys": [],
                          "seg": [],
                          "start_state": [],
                          "end_state": [],
                          "start": [],
                          "goal": [],
                          "var_num": [],
                          "eq_num": [],
                          "ineq_num1": [],
                          "ineq_num2": [],
                          "ineq_num": [],
                          "inner_pts": [],
                          "waypts": [],
                          "path_length": [],
                          "time_lb": [],
                          "init_time_factor": [],
                          "Times: ": [],
                          "params": [],
                          "ref_time_factor": []
                          }

        stacked_state = stacked_state.float()
        stacked_hpolys = stacked_hpolys.float()

        state_embeddings = self.state_input_module(stacked_state)
        hpoly_embeddings = self.hpoly_input_module(stacked_hpolys)

        state_embeddings = state_embeddings.squeeze(-1)
        hpoly_embeddings = hpoly_embeddings.squeeze(-1)

        combined = torch.cat((state_embeddings, hpoly_embeddings), dim=1).type(torch.float32)

        print("The dimension of combined is : ", combined.shape)

        # For LSTM
        tfs = None
        # print("shape of ts is : ", ts.shape)

        all_stop_tokens = None

        for j in range(combined.shape[0]):
            tf = None
            stop_tokens = None
            h = torch.zeros(1, self.hidden_size).to(self.device)
            c = torch.zeros(1, self.hidden_size).to(self.device)

            curr_input = combined[j].unsqueeze(0)
            for k in range(5):
                lstm_out, (h, c) = self.output_module(curr_input, (h, c))

                curr_tf_elem = self.tfs_output_layer(lstm_out)
                if tf is None:
                    tf = curr_tf_elem
                else:
                    tf = torch.cat((tf, curr_tf_elem), dim=0)

                curr_stop_tokens = self.stop_token_output_layer(lstm_out)

                if stop_tokens is None:
                    stop_tokens = curr_stop_tokens
                else:
                    stop_tokens = torch.cat((stop_tokens, curr_stop_tokens), dim=0)

                if curr_stop_tokens.item() > 0.5:
                    break

            # Check if stop_tokens has 5 elements
            print("shape of stop_tokens is : ", stop_tokens.shape)
            print("shape of tf is : ", tf.shape)
            print("stop_tokens is : ", stop_tokens)
            print("tf is : ", tf)
            # Change the first and second dimension
            stop_tokens = stop_tokens.permute(1, 0)
            tf = tf.permute(1, 0)
            print("shape of stop_tokens is (after permute) : ", stop_tokens.shape)
            print("shape of tf is (after permute) : ", tf.shape)
            print("stop_tokens is (after permute) : ", stop_tokens)
            print("tf is (after permute) : ", tf)
            if stop_tokens.shape[1] < 5:
                #print("shape of stop_tokens is : ", stop_tokens.shape)
                #print("shape of torch.ones(5 - stop_tokens.shape[0]).to(self.device) is : ",  torch.ones(5 - stop_tokens.shape[0]).to(self.device).shape)
                # Pad with 1
                stop_tokens_padding = torch.ones(5 - stop_tokens.shape[1])
                stop_tokens_padding = stop_tokens_padding.unsqueeze(0)
                stop_tokens = torch.cat((stop_tokens, stop_tokens_padding.to(self.device)), dim=1)
                #print("shape of stop_tokens is (<5) : ", stop_tokens.shape)
            #else:
                #print("shape of stop_tokens is (>=5) : ", stop_tokens.shape)

            # Check if tf has 5 elements
            if tf.shape[1] < 5:
                # Pad with 0
                tf_padding = torch.zeros(5 - tf.shape[1])
                tf_padding = tf_padding.unsqueeze(0)
                tf = torch.cat((tf, tf_padding.to(self.device)), dim=1)

            print("Current tf is : ", tf)
            print("Current stop_tokens is : ", stop_tokens)

            if tfs is None:
                tfs = tf
                all_stop_tokens = stop_tokens
            else:
                tfs = torch.cat((tfs, tf), dim=0)
                all_stop_tokens = torch.cat((all_stop_tokens, stop_tokens), dim=0)

            print("Shape of tfs is : ", tfs.shape)
            print("Shape of all_stop_tokens is : ", all_stop_tokens.shape)

        obj_values = []
        obj1_values = []
        objt_values = []
        objc_values = []
        stop_token_loss_values = []


        osqplayer = OsqpLayer()

        qp_traj = MinTrajOpt([])

        num_has_solution = 0

        num_time_segment_accurate = 0

        num_time_segment_accurate_if_has_solution = 0

        for i in range(tfs.shape[0]):
            tf = tfs[i].cpu()
            x_i = stacked_state[i].cpu()
            hpoly_i = stacked_hpolys[i].cpu()
            traj_times_i = traj_times[i].cpu()

            stop_tokens_i = all_stop_tokens[i].cpu()
            
            qp_traj.update(x_i, hpoly_i, tf, phase=self.phase, traj_times=traj_times_i)

            ref_tf = qp_traj.ref_time_factor

            tf = tf.type(torch.float32)
            ref_tf = ref_tf.type(torch.float32)

            curr_objt_val = nn.MSELoss()(tf, ref_tf)


            curr_obj_value = curr_objt_val

            objc_values.append(torch.tensor(0.0))
            obj_values.append(curr_obj_value)
            obj1_values.append(torch.tensor(0.0))
            objt_values.append(curr_objt_val)

            stop_token_loss_values.append(torch.tensor(0.0))

            del x_i
            del hpoly_i

            gc.collect()
            torch.cuda.empty_cache()

        if self.loss_dim_reduction == "mean":
            obj_values_stack = torch.stack(obj_values)
            obj_mask = obj_values_stack != 0
            if torch.sum(obj_mask) > 0:  # Check if there's any non-zero elements
                obj_value = torch.mean(obj_values_stack[obj_mask])
            else: 
                print("All elements are zero!")
                obj_value = torch.tensor(0.0)
            
            obj1_val_stack = torch.stack(obj1_values)
            obj1_mask = obj1_val_stack != 0
            if torch.sum(obj1_mask) > 0: 
                obj1_val = torch.mean(obj1_val_stack[obj1_mask])
            else:
                obj1_val = torch.tensor(0.0)

            
            objt_val = torch.mean(torch.stack(objt_values))

            objc_val = torch.mean(torch.stack(objc_values))

            stop_token_loss_val = torch.mean(torch.stack(stop_token_loss_values))

            # obj4_val = torch.mean(torch.stack(obj4_values))
        elif self.loss_dim_reduction == "sum":
            obj_value = torch.sum(torch.stack(obj_values))
            obj1_val = torch.sum(torch.stack(obj1_values))
            objt_val = torch.sum(torch.stack(objt_values))
            objc_val = torch.sum(torch.stack(objc_values))

            stop_token_loss_val = torch.sum(torch.stack(stop_token_loss_values))
        else:
            raise ValueError("loss_dim_reduction must be either mean or sum")


        del combined
        del qp_traj
        del osqplayer

        gc.collect()
        torch.cuda.empty_cache()

        print("====== Number of successful solutions: ", num_has_solution, " ======")

        success_rate = num_has_solution / tfs.shape[0]

        percent_time_segment_accurate = num_time_segment_accurate / tfs.shape[0]
        if num_has_solution == 0:
            percent_time_segment_accurate_if_has_solution = 0
        else:
            percent_time_segment_accurate_if_has_solution = num_time_segment_accurate_if_has_solution / num_has_solution

        print("====== Success rate: ", success_rate, " ======")
        print("====== Percent of time segments accurate: ", percent_time_segment_accurate, " ======")
        print("====== Percent of time segments accurate if has solution: ", percent_time_segment_accurate_if_has_solution, " ======")

        # Ensure all return values are on the same device
        obj_value = obj_value.to(self.device)
        obj1_val = obj1_val.to(self.device)
        objt_val = objt_val.to(self.device)

        objc_val = objc_val.to(self.device)

        # obj4_val = obj4_val.to(self.device)

        return obj_value, obj1_val, objt_val, objc_val, success_rate, objt_debug_log, stop_token_loss_val
    
    def train_model(self, stacked_state, stacked_hpolys, traj_times):
        self.optimizer.zero_grad()

        print("stacked_state.dtype: ", stacked_state.dtype)
        print("stacked_hpolys.dtype: ", stacked_hpolys.dtype)
        print("traj_times.dtype: ", traj_times.dtype)

        #for param in self.parameters():
            #print("param.dtype: ", param.dtype)

        # convert obj_value to a loss
        obj_val, obj1_val, objt_val, objc_val, success_rate, objt_debug_log, stop_token_loss_val =\
              self.forward_batch(stacked_state, stacked_hpolys, traj_times)

        # Check if obj_val has gradient
        if obj_val.requires_grad:
            obj_val.backward()

            self.optimizer.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()



        # Clone the values before deletion if they are needed after this function call
        obj_val_clone = obj_val.detach().item()
        obj1_val_clone = obj1_val.detach().item()

        objt_val_clone = objt_val.detach().item()

        objc_val_clone = objc_val.detach().item()

        stop_token_loss_val_clone = stop_token_loss_val.detach().item()


        del obj_val, obj1_val, objt_val
        gc.collect()
        torch.cuda.empty_cache()

        # Return the cloned values
        return obj_val_clone, obj1_val_clone, objt_val_clone, objc_val_clone, success_rate, objt_debug_log, stop_token_loss_val_clone

    def eval_model(self, stacked_state, stacked_hpolys, traj_times):
        obj_val, obj1_val, objt_val, objc_val = self.forward_batch(stacked_state, stacked_hpolys, traj_times)
        return obj_val, obj1_val, objt_val, objc_val
