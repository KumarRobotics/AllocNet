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




class MinimalSnapNetwork(nn.Module):
    def __init__(self, seg, max_poly_length, lr=1e-3,
                 T_0=500, T_mult=1, eta_min=1e-5, last_epoch=-1, hidden_size=128, use_scheduler=True,
                 loss_dim_reduction='mean', w1=1, w2=100000, wt= 5.0, poly_mode='vpoly'):
        super().__init__()
        #init
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg = seg

        self.output_module = nn.Sequential(
            nn.Linear(12, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, self.seg),
            nn.Softplus(beta=2, threshold=5)
        )

        for layer in self.output_module:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.output_module.to(self.device)

        self.max_poly_length = max_poly_length

        if poly_mode == 'vpoly':
            self.input_module = nn.Sequential(
                nn.Conv1d(self.max_poly_length, 32, kernel_size=3, dtype=float),
                nn.ReLU(),
                nn.Conv1d(32, 6, kernel_size=1, dtype=float)
            )
        elif poly_mode == 'hpoly':
            self.input_module = nn.Sequential(
                nn.Conv1d(self.max_poly_length, 32, kernel_size=4, dtype=float),
                nn.ReLU(),
                nn.Conv1d(32, 6, kernel_size=1, dtype=float)
            )
        else:
            raise ValueError("poly_mode must be either vpoly or hpoly")
        self.input_module.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=T_0,
                                                                            T_mult=T_mult,
                                                                            eta_min=eta_min,
                                                                            last_epoch=last_epoch)

        self.loss_dim_reduction = loss_dim_reduction

        self.w1 = w1
        self.w2 = w2
        # self.w3 = w3
        self.wt = wt

        #self.w4 = w4

        self.to(self.device)

        self.poly_mode = poly_mode

        self.phase = 1

    def forward(self, x, hpoly_elems, hpoly_end_probs):
        torch.no_grad()
        #diff = x[:, 9:18] - x[:, 0:9]
        new_x  = torch.hstack([x[:, 0:3], x[:, 9:12]])
        time_network_start_time = time.time()

        b = self.input_module(hpoly_elems)
        b = b.squeeze(-1)

        combined = torch.cat((new_x, b), dim=1).type(torch.float32)

        ts = self.output_module(combined)

        time_network_end_time = time.time()

        print("Duration for time network: ", time_network_end_time - time_network_start_time)
        """
        obj = lambda z, Csqrt, G, h, A, b: cp.sum_squares(Csqrt @ z) if isinstance(z, cp.Variable) else torch.sum(
            (Csqrt @ z) ** 2)
        ineq = lambda z, Csqrt, G, h, A, b: G @ z - h
        s_eq = lambda z, Csqrt, G, h, A, b: A @ z - b

        vars = [cp.Variable(self.num)]
        params = [cp.Parameter((self.num, self.num)),
                       cp.Parameter((self.ineq_num, self.num)), cp.Parameter(self.ineq_num),
                       cp.Parameter((self.eq_num, self.num)), cp.Parameter(self.eq_num)]

        cp_eq = [eq(*vars, *params) == 0 for eq in [s_eq]]
        cp_ineq = [eq(*vars, *params) <= 0 for eq in [ineq]]

        problem = cp.Problem(cp.Minimize(obj(*vars, *params)), cp_ineq + cp_eq)

        cvxpylayer = CvxpyLayer(problem, parameters=params, variables=vars)
        cvxpylayer.to(self.device)
        """
        osqplayer = OsqpLayer()
        qp_traj = MinTrajOpt([])

        i = 0
        t = ts[i].cpu()
        print("t: ", t)
        x_i = x[i].cpu()
        hpoly_elems_i = hpoly_elems[i].cpu()
        hpoly_end_probs_i = hpoly_end_probs[i].cpu()

        qp_traj.update(x_i, hpoly_elems_i, hpoly_end_probs_i, t)

        """
        relax_params = [params[0].cpu() + 0.001 * torch.eye(self.num),
                        params[1].cpu(), params[1].mv(self.z0.cpu()) + torch.exp(self.s0.cpu()),
                        params[3].cpu(), params[3].mv(self.z0.cpu())]
        
        solution, = cvxpylayer(*relax_params)

        print("solution: ", solution)
        print("cvxpylayer.info: ", cvxpylayer.info)
        """

        #qp_traj.solve()

        solution, _, _, _, _ = osqplayer(qp_traj)


        return solution

    # x includes start-end state and hpolys
    # @profile
    def forward_batch(self, x, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs):
        #tracemalloc.start()
        ##### NN for time allocation #####
        #diff = x[:, 9:18] - x[:, 0:9]
        # diff = x
        new_x  = torch.hstack([x[:, 0:3], x[:, 9:12]])


        if self.poly_mode == 'vpoly':
            b = self.input_module(vpoly_elems)
        elif self.poly_mode == 'hpoly':
            b = self.input_module(hpoly_elems)
        else:
            raise ValueError("poly_mode must be either vpoly or hpoly")
        b = b.squeeze(-1)

        combined = torch.cat((new_x, b), dim=1).type(torch.float32)


        ts = self.output_module(combined)
        # print("shape of ts is : ", ts.shape)

        obj_values = []
        obj1_values = []
        obj2_values = []
        #obj3_values = []
        objt_values = []

        osqplayer = OsqpLayer()

        qp_traj = MinTrajOpt([])
        # print("------ MEMORY USAGE: forward_batch before loop: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, " MB ------")
        for i in range(ts.shape[0]):
            t = ts[i].cpu()
            x_i = x[i].cpu()
            hpoly_elems_i = hpoly_elems[i].cpu()
            hpoly_end_probs_i = hpoly_end_probs[i].cpu()

            qp_traj.update(x_i, hpoly_elems_i, hpoly_end_probs_i, t, phase=self.phase)

            res = osqplayer(qp_traj)


            if res != None:


                solution, curr_obj1_val, curr_objt_val = res
                print("curr_objt_value is : ",   curr_objt_val)
                #curr_objt_val = torch.clamp(new_time, min=t_lower_bound)

                curr_obj2_val = torch.tensor(0.0)

                curr_obj_value = curr_obj1_val*self.w1 + curr_objt_val*self.wt
                #curr_obj_value = curr_obj1_val*self.w1
                
                print("curr_obj1_val is : ", curr_obj1_val)
                print("curr_obj2_val is : ", curr_obj2_val)
                print("curr_obj_value is : ", curr_obj_value)
                print("curr_objt_value is : ",  curr_objt_val)


                obj_values.append(curr_obj_value)
                obj1_values.append(curr_obj1_val)
                obj2_values.append(curr_obj2_val)
                #obj3_values.append(curr_obj3_val)
                objt_values.append(curr_objt_val)
            del t
            del x_i
            del hpoly_elems_i
            del hpoly_end_probs_i

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
            
            obj2_val_stack = torch.stack(obj2_values)
            obj2_mask = obj2_val_stack != 0
            if torch.sum(obj2_mask) > 0:
                obj2_val = torch.mean(obj2_val_stack[obj2_mask])
            else:
                obj2_val = torch.tensor(0.0)
            
            
            objt_val = torch.mean(torch.stack(objt_values))

            # obj4_val = torch.mean(torch.stack(obj4_values))
        elif self.loss_dim_reduction == "sum":
            obj_value = torch.sum(torch.stack(obj_values))
            obj1_val = torch.sum(torch.stack(obj1_values))
            obj2_val = torch.sum(torch.stack(obj2_values))
            #obj3_val = torch.sum(torch.stack(obj3_values))
            objt_val = torch.sum(torch.stack(objt_values))

            # obj4_val = torch.sum(torch.stack(obj4_values))
        else:
            raise ValueError("loss_dim_reduction must be either mean or sum")

 
        #del diff
        del b
        del combined
        # del cvxpylayer
        del qp_traj
        # del problem
        # del cp_eq
        # del cp_ineq
        #del vars
        del osqplayer
        # del s_eq
        # del ineq
        # del obj

        gc.collect()
        torch.cuda.empty_cache()

        # Ensure all return values are on the same device
        obj_value = obj_value.to(self.device)
        obj1_val = obj1_val.to(self.device)
        obj2_val = obj2_val.to(self.device)
        #obj3_val = obj3_val.to(self.device)
        objt_val = objt_val.to(self.device)

        # obj4_val = obj4_val.to(self.device)

        return obj_value, obj1_val, obj2_val, objt_val
    
    def train_model(self, x, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs):
        # print("------ MEMORY USAGE: train_model before zero_grad: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, " MB ------")
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()

        # convert obj_value to a loss
        obj_val, obj1_val, obj2_val, objt_val = self.forward_batch(x, hpoly_elems, hpoly_end_probs, vpoly_elems, vpoly_end_probs)

        # Check if obj_val has gradient
        if obj_val.requires_grad:
            obj_val.backward()

            self.optimizer.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

        # Clone the values before deletion if they are needed after this function call
        obj_val_clone = obj_val.detach().item()
        obj1_val_clone = obj1_val.detach().item()
        obj2_val_clone = obj2_val.detach().item()
        #obj3_val_clone = obj3_val.detach().item()
        objt_val_clone = objt_val.detach().item()

        # obj4_val_clone = obj4_val.detach().item()
        # obj_val_clone = obj_val.clone()  # for non-scalar tensors

        del obj_val, obj1_val, objt_val
        gc.collect()
        torch.cuda.empty_cache()

        # local_vars = list(locals().items())
        # for var, obj in local_vars:
        #     print(var, sys.getsizeof(obj))

        # Return the cloned values
        return obj_val_clone, obj1_val_clone, obj2_val_clone, objt_val_clone

    def eval_model(self, x, hpoly_elems, hpoly_end_probs):
        obj_val, obj1_val, obj2_val, objt_val = self.forward_batch(x, hpoly_elems, hpoly_end_probs)
        return obj_val, obj1_val, obj2_val, objt_val
