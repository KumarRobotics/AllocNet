import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from itertools import accumulate
import cvxpy as cp

import math
import time
from scipy import sparse
import osqp
import numpy as np



class Relu(nn.Module):
    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nFeatures)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)

    def __call__(self, x):
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = self.fc2(x)
        return x
    


class OsqpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.prob = osqp.OSQP()

        self.obj  = lambda z,Q,A,b,G,h: 0.5 * z@Q@z
        self.eq   = lambda z,Q,A,b,G,h: A@z - b
        self.ineq = lambda z,Q,A,b,G,h: G@z - h

        self.ineq_vio = lambda z,G,h: G@z - h

        self.s0 = nn.Parameter(torch.randn(5))
        self.pos_s0 = nn.Parameter(torch.randn(2000))

    
    def forward(self, qp_traj):
       
        parameters = qp_traj.params
        init_time_factor = qp_traj.Times

        print("init_time_factor is", init_time_factor)
        # Check if qp_traj have the attribute ref_time_factor
        if hasattr(qp_traj, 'ref_time_factor'):
            print("qp_traj.ref_time_factor is", qp_traj.ref_time_factor)


        self.var_num = parameters[0].shape[0]
        P = sparse.csc_matrix(parameters[0].detach().numpy())
        q = np.zeros(self.var_num)

        ########################### call the solver
        self.eq_num = parameters[1].shape[0]
        self.ineq_num = parameters[4].shape[0] + parameters[6].shape[0]
        params = [parameters[0],
                  parameters[1],  parameters[2],
                  torch.vstack([parameters[3], parameters[5]]), 
                  torch.hstack([parameters[4], parameters[6]])]

        A = sparse.vstack([params[1].detach().numpy(), params[3].detach().numpy()], format='csc')
        u = np.hstack([params[2].detach().numpy(),  params[4].detach().numpy()])
        l = np.hstack([params[2].detach().numpy(), -math.inf * np.ones(self.ineq_num)])
        prob = osqp.OSQP()

        prob.setup(P, q, A, l, u, verbose=False, warm_start=True)
        #prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        segments = qp_traj.seg

        curr_obj1_val = torch.sum(init_time_factor[0: segments]) /(1.0 * segments)
        #print("curr_obj1_val is", curr_obj1_val)

        zero_segments = 5 - segments

        print("zero_segments is", zero_segments)

        if zero_segments != 0:
            curr_padding_loss = nn.MSELoss()(init_time_factor[segments:], torch.zeros(zero_segments, dtype=torch.float64))
        else:
            curr_padding_loss = torch.tensor(0.0, dtype=torch.float64)

        print("curr_padding_loss is", curr_padding_loss)

        if res.info.status != 'solved':


            curr_objt_val = None
            if hasattr(qp_traj, 'ref_time_factor'):
                # Check at which element ref_time_factor is 0.0000
                curr_objt_val = nn.MSELoss()(init_time_factor[0:segments], qp_traj.ref_time_factor[0:segments]) / (
                            1.0 * segments)

                if zero_segments != 0:
                    curr_padding_loss = nn.MSELoss()(init_time_factor[segments:],
                                                     torch.zeros(zero_segments, dtype=torch.float64))
                else:
                    curr_padding_loss = torch.tensor(0.0, dtype=torch.float64)

                curr_objt_val = curr_objt_val + curr_padding_loss

            return None, curr_obj1_val , curr_objt_val, None, curr_padding_loss
        else:
 
            print("======================== solved ===================")
            z = torch.tensor(res.x , dtype=float).detach().requires_grad_()

            lam = torch.tensor(res.y[self.eq_num: ], dtype=float)
            nu  = torch.tensor(res.y[0:self.eq_num], dtype=float)

            # compute residuals and re-engage autograd tape
            y = torch.hstack((z, lam, nu))
            

            g = self.ineq(z, *params)
            # var_num + ineq + eq 
            J_1 = torch.hstack((params[0], params[3].transpose(0,1)@torch.diag(lam), params[1].transpose(0,1)))
            # ineq (var, ineq,  eq)
            J_2 = torch.hstack((params[3],torch.diag(g), torch.zeros((self.ineq_num, self.eq_num), dtype=float)))
            J_3 = torch.hstack((params[1], torch.zeros((self.eq_num, self.ineq_num+self.eq_num), dtype=float)))
            J =  torch.vstack((J_1, J_2, J_3))

            def get_grad(grad):
                
                #print("======= grad is", grad)
                if torch.linalg.matrix_rank(J) == J.shape[0]:
                    #print("new grad is", torch.linalg.solve(J, grad))
                    grad[:] = -torch.linalg.solve(J, grad)  
                else: 
                    #print("new grad is", torch.linalg.pinv(J) @ grad)
                    grad[:] = -torch.linalg.pinv(J) @ grad
 
            y.register_hook(get_grad)

            curr_objc_val = self.obj(y[0:self.var_num], *params)/qp_traj.path_length

            return y[0:self.var_num], curr_obj1_val , None, curr_objc_val, curr_padding_loss

    def forward4lstm(self, qp_traj, pred_stop_tokens, seq_len=5):

        parameters = qp_traj.params
        init_time_factor = qp_traj.Times

        print("init_time_factor is", init_time_factor)
        # Check if qp_traj have the attribute ref_time_factor
        if hasattr(qp_traj, 'ref_time_factor'):
            print("qp_traj.ref_time_factor is", qp_traj.ref_time_factor)

        self.var_num = parameters[0].shape[0]
        P = sparse.csc_matrix(parameters[0].detach().numpy())
        q = np.zeros(self.var_num)

        ########################### call the solver
        self.eq_num = parameters[1].shape[0]
        self.ineq_num = parameters[4].shape[0] + parameters[6].shape[0]
        params = [parameters[0],
                  parameters[1], parameters[2],
                  torch.vstack([parameters[3], parameters[5]]),
                  torch.hstack([parameters[4], parameters[6]])]

        A = sparse.vstack([params[1].detach().numpy(), params[3].detach().numpy()], format='csc')
        u = np.hstack([params[2].detach().numpy(), params[4].detach().numpy()])
        l = np.hstack([params[2].detach().numpy(), -math.inf * np.ones(self.ineq_num)])
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, warm_start=True)
        # prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()

        segments = qp_traj.seg

        curr_obj1_val = torch.sum(init_time_factor[0: segments]) / (1.0 * segments)
        # print("curr_obj1_val is", curr_obj1_val)

        segments_after_end = seq_len - segments

        gt_stop_tokens_before_end = torch.zeros(segments - 1)
        gt_stop_tokens_after_end = torch.ones(segments_after_end + 1)

        gt_stop_tokens = torch.cat((gt_stop_tokens_before_end, gt_stop_tokens_after_end), 0)

        # Calculate premature end penalty
        end_penalty= 5.0
        token_thresh = 0.42
        premature_end_mask = (pred_stop_tokens > token_thresh) & (gt_stop_tokens < token_thresh)
        premature_end_penalty = premature_end_mask.float().sum() * end_penalty

        late_end_mask = (pred_stop_tokens < token_thresh) & (gt_stop_tokens > token_thresh)
        late_end_penalty = late_end_mask.float().sum() * end_penalty

        #stop_token_loss = nn.BCELoss()(pred_stop_tokens, gt_stop_tokens) + premature_end_penalty
        stop_token_loss = nn.BCELoss()(pred_stop_tokens, gt_stop_tokens) + premature_end_penalty + late_end_penalty

        if res.info.status != 'solved':

            curr_objt_val = None
            if hasattr(qp_traj, 'ref_time_factor'):
                # Check at which element ref_time_factor is 0.0000
                curr_objt_val = nn.MSELoss()(init_time_factor[0:segments], qp_traj.ref_time_factor[0:segments]) / (
                        1.0 * segments)


            return None, curr_obj1_val, curr_objt_val, None, stop_token_loss
        else:

            print("======================== solved ===================")
            z = torch.tensor(res.x, dtype=float).detach().requires_grad_()

            lam = torch.tensor(res.y[self.eq_num:], dtype=float)
            nu = torch.tensor(res.y[0:self.eq_num], dtype=float)

            # compute residuals and re-engage autograd tape
            y = torch.hstack((z, lam, nu))

            g = self.ineq(z, *params)
            # var_num + ineq + eq
            J_1 = torch.hstack((params[0], params[3].transpose(0,1)@torch.diag(lam), params[1].transpose(0,1)))
            # ineq (var, ineq,  eq)
            J_2 = torch.hstack((params[3],torch.diag(g), torch.zeros((self.ineq_num, self.eq_num), dtype=float)))
            J_3 = torch.hstack((params[1], torch.zeros((self.eq_num, self.ineq_num+self.eq_num), dtype=float)))
            J =  torch.vstack((J_1, J_2, J_3))

            def get_grad(grad):
                
                if torch.linalg.matrix_rank(J) == J.shape[0]:
                    grad[:] = -torch.linalg.solve(J, grad)  
                else: 
                    grad[:] = -torch.linalg.pinv(J) @ grad
 
            y.register_hook(get_grad)

            curr_objc_val = self.obj(y[0:self.var_num], *params) / qp_traj.path_length

            return y[0:self.var_num], curr_obj1_val, None, curr_objc_val, stop_token_loss