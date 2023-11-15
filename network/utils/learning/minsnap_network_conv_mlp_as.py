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




class ConvMLPMinimalSnapNetwork4AblationStudy(nn.Module):
    def __init__(self, seg, max_poly_length, lr=1e-3,
                 T_0=500, T_mult=1, eta_min=1e-5, last_epoch=-1, hidden_size=128, use_scheduler=True,
                 loss_dim_reduction='mean', w1=1, wt= 5.0, wc=1.0, wp=1.0):
        super().__init__()
        #init

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg = seg

        self.output_module = nn.Sequential(
            nn.Linear(38, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_size, self.seg),
            nn.Softplus(beta=2)
        )

        for layer in self.output_module:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.output_module.to(self.device)

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
        osqplayer = OsqpLayer()
        qp_traj = MinTrajOpt([])

        inference_start_time = time.time()

        stacked_state = stacked_state.float()
        stacked_hpolys = stacked_hpolys.float()

        state_embeddings = self.state_input_module(stacked_state)
        hpoly_embeddings = self.hpoly_input_module(stacked_hpolys)

        state_embeddings = state_embeddings.squeeze(-1)
        hpoly_embeddings = hpoly_embeddings.squeeze(-1)

        combined = torch.cat((state_embeddings, hpoly_embeddings), dim=1).type(torch.float32)

        tfs = self.output_module(combined)

        inference_end_time = time.time()

        print("Inference time is: ", inference_end_time - inference_start_time)

        tf = tfs[0].cpu()
        x_i = stacked_state[0].cpu()
        hpoly_i = stacked_hpolys[0].cpu()
        traj_times_i = None

        qp_traj.update(x_i, hpoly_i, tf, phase=self.phase, traj_times=traj_times_i)

        solution, curr_obj1_val, curr_objt_val, curr_objc_val, curr_padding_loss = osqplayer(qp_traj)

        return solution, curr_obj1_val, curr_objt_val, curr_objc_val

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

        tfs = self.output_module(combined)

        print("shape of tfs is : ", tfs.shape)

        obj_values = []
        obj1_values = []
        objt_values = []
        objc_values = []
        padding_loss_values = []


        osqplayer = OsqpLayer()

        qp_traj = MinTrajOpt([])

        num_has_solution = 0

        for i in range(tfs.shape[0]):
            tf = tfs[i].cpu()
            x_i = stacked_state[i].cpu()
            hpoly_i = stacked_hpolys[i].cpu()
            traj_times_i = traj_times[i].cpu()
            
            qp_traj.update(x_i, hpoly_i, tf, phase=self.phase, traj_times=traj_times_i)
            # res = osqplayer(qp_traj)


            solution, curr_obj1_val, curr_objt_val, curr_objc_val, curr_padding_loss = osqplayer(qp_traj)
            #print("curr_objt_val: ", curr_objt_val)
            if solution != None:
                num_has_solution += 1
                # if curr_objc_val is not None:
                #     objc_values.append(curr_objc_val)
                #     curr_obj_value = curr_obj1_val*self.w1 + curr_objt_val*self.wt + curr_objc_val * self.wc
                # else:
                #     curr_obj_value = curr_obj1_val*self.w1 + curr_objt_val*self.wt


                curr_obj_value = curr_obj1_val*self.w1 + curr_objc_val * self.wc + curr_padding_loss*self.wp

                objc_values.append(curr_objc_val)
                obj_values.append(curr_obj_value)
                obj1_values.append(curr_obj1_val)
                objt_values.append(torch.tensor(0.0))

                padding_loss_values.append(curr_padding_loss)
                
            else:

                curr_obj_value = curr_obj1_val*self.w1 + curr_objt_val*self.wt + curr_padding_loss*self.wp

                objc_values.append(torch.tensor(0.0))
                obj_values.append(curr_obj_value)
                obj1_values.append(curr_obj1_val)
                objt_values.append(curr_objt_val)

                padding_loss_values.append(curr_padding_loss)


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

            padding_loss_val = torch.mean(torch.stack(padding_loss_values))

            # obj4_val = torch.mean(torch.stack(obj4_values))
        elif self.loss_dim_reduction == "sum":
            obj_value = torch.sum(torch.stack(obj_values))
            obj1_val = torch.sum(torch.stack(obj1_values))
            objt_val = torch.sum(torch.stack(objt_values))
            objc_val = torch.sum(torch.stack(objc_values))

            padding_loss_val = torch.sum(torch.stack(padding_loss_values))
        else:
            raise ValueError("loss_dim_reduction must be either mean or sum")


        del combined
        del qp_traj
        del osqplayer

        gc.collect()
        torch.cuda.empty_cache()

        print("====== Number of successful solutions: ", num_has_solution, " ======")

        success_rate = num_has_solution / tfs.shape[0]

        # Ensure all return values are on the same device
        obj_value = obj_value.to(self.device)
        obj1_val = obj1_val.to(self.device)
        objt_val = objt_val.to(self.device)

        objc_val = objc_val.to(self.device)

        # obj4_val = obj4_val.to(self.device)

        return obj_value, obj1_val, objt_val, objc_val, success_rate, objt_debug_log, padding_loss_val
    
    def train_model(self, stacked_state, stacked_hpolys, traj_times):
        self.optimizer.zero_grad()

        print("stacked_state.dtype: ", stacked_state.dtype)
        print("stacked_hpolys.dtype: ", stacked_hpolys.dtype)
        print("traj_times.dtype: ", traj_times.dtype)

        #for param in self.parameters():
            #print("param.dtype: ", param.dtype)

        # convert obj_value to a loss
        obj_val, obj1_val, objt_val, objc_val, success_rate, objt_debug_log, padding_loss_val =\
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

        padding_loss_val_clone = padding_loss_val.detach().item()


        del obj_val, obj1_val, objt_val
        gc.collect()
        torch.cuda.empty_cache()

        # Return the cloned values
        return obj_val_clone, obj1_val_clone, objt_val_clone, objc_val_clone, success_rate, objt_debug_log, padding_loss_val_clone

    def eval_model(self, stacked_state, stacked_hpolys, traj_times):
        obj_val, obj1_val, objt_val, objc_val = self.forward_batch(stacked_state, stacked_hpolys, traj_times)
        return obj_val, obj1_val, objt_val, objc_val
