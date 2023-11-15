import numpy as np
import scipy
import cvxpy as cp
from utils.trajectory import Trajectory
from math import sqrt
import torch.autograd as autograd
import torch
import math
import time
from scipy.sparse import csc_matrix
import yaml
import osqp
import scipy as sp
from scipy import sparse

from memory_profiler import profile


class MinTrajOpt:

    def __init__(self, params):
        
        #print("[MinTrajOpt] The state is :", state)
        if params == []:
                        
            with open("utils/params.yaml", "r") as stream:
                try:
                    self.params = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            self.params = params
        #dynamic feasibility
        # trajectory porperties
        self.order = self.params['planning']['order']
        self.state_dim = self.params['planning']['state_dim'] 
        self.dim = self.params['planning']['dim'] 
        self.res = self.params['planning']['res']

        self.D =  2 * self.order
        self.use_time_factor = self.params['planning']['use_time_factor']

        self.phy_limits = torch.zeros(3)
        self.phy_limits[0] = self.params['physical_limits']['max_vel']
        self.phy_limits[1] = self.params['physical_limits']['max_acc']
        self.phy_limits[2] = self.params['physical_limits']['max_jerk']



        self.phase1_phy_limits = torch.zeros(4)
        self.phase1_phy_limits[0] = self.params['phase1_physical_limits']['max_vel']
        self.phase1_phy_limits[1] = self.params['phase1_physical_limits']['max_acc']
        self.phase1_phy_limits[2] = self.params['phase1_physical_limits']['max_jerk']
        self.phase1_phy_limits[3] = self.params['phase1_physical_limits']['inf_dis']

        if self.order == 3: 
            self.zero_A = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                                        [0.0, 0.0, 6.0, 0.0, 0.0, 0.0]], dtype=float)
        else:
            self.zero_A = torch.tensor([[0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]], dtype=float)


    def update(self, state, hpolys, time_factor, phase=1, traj_times=None, seq_len=5):

        self.hpolys = []
        #start_idx = 0

        const_num = 0

        for i in range(seq_len):
            #print(hpolys[i])
            poly = hpolys[:, :, i].type(torch.float64)

            if torch.norm(hpolys[:, :, i]) <= 1.0:
                break

            #print(poly.shape[0])
            for j in range(poly.shape[0]):
                if torch.norm(poly[j, :]) <= 0.0:
                    poly = poly[0:j, :]
                    break


            self.hpolys.append(poly)
            const_num += poly.shape[0]

        self.seg = len(self.hpolys)
        #print("===========self.seg", self.seg)

 
        #setz zero
        self.start_state = state[:, 0]
        self.end_state   = state[:, 1]

        self.start = [self.start_state[0], self.start_state[3], self.start_state[6]]
        self.goal = [self.end_state[0], self.end_state[3], self.end_state[6]]

        #print("seg is ", self.seg)
        self.var_num = self.seg * self.dim * self.D
        #print("self.var_num is ", self.var_num)

        self.eq_num = (2 * self.state_dim + (self.order ) * (self.seg-1) ) * self.dim
        self.ineq_num1 = self.res * const_num
        self.ineq_num2 = self.res * 4 * self.dim * self.seg
        self.ineq_num = self.ineq_num1 + self.ineq_num2


        if self.use_time_factor: 

            self.inner_pts = self.get_inner_pts()  # refine goal position
            self.waypts = np.vstack((self.start, self.inner_pts, self.goal))
            print(self.waypts)

            self.path_length = 0
            for i in range(len(self.waypts) - 1):

                self.path_length += np.linalg.norm(self.waypts[i+1] - self.waypts[i])
     
    

            self.time_lb = self.getT_lbs(self.waypts,self.phy_limits[0], self.phy_limits[1])        
         
            # if time_factor == []: 

            #     time_factor  = self.initT(self.waypts,self.phy_limits[0], self.phy_limits[1])
            #self.init_time_factor = time_factor.type(torch.float64)

            self.Times = self.time_lb  +  self.time_lb * time_factor.type(torch.float64)


            if traj_times != None:
                self.ref_time_factor = traj_times.type(torch.float64) / self.time_lb



        else:
            #self.init_time_factor = time_factor.type(torch.float64)
            
            self.Times = time_factor.type(torch.float64)
            self.path_length = np.linalg.norm(np.array(self.goal) -  np.array(self.start))


            if traj_times != None:
                self.ref_time_factor = traj_times.type(torch.float64)




        # print("time_factor is", time_factor)

        Q, A, b = self.fill_eq_obj(self.Times)

        #compute matrix
        if phase == 1:
            G1, h1, G2, h2 = self.fill_phase1_ineq(self.Times)
        else:
            G1, h1, G2, h2 = self.fill_ineq(self.Times)

        # print("Times is ", Times)

        # print(Q.shape)
        # print(A.shape)
        # print(b.shape)
        # print(G1.shape)
        # print(h1.shape)
        # print(G2.shape)
        # print(h2.shape)
        self.params = [Q, A, b, G1, h1, G2, h2]

        #print(traj_times)


        return





    # using trapezoidal velocity profile
    def initT(self, pts, maxv, maxa):
        n = pts.shape[0] - 1
        times = torch.zeros(self.seg)

        for i in range(n):
            times[i] = 1.0
        return times
    
    

    def getT_lbs(self, pts, maxv, maxa):


        n = pts.shape[0] - 1
        times = torch.zeros(5)

        for i in range(n):
            dis = pts[i+1] - pts[i]
            vel_t = abs(dis/maxv)
            acc_t = abs(2 * dis / maxa)

            times[i] = max(vel_t.max(), sqrt(acc_t.max()))

            #print("time is:", times[i])

        return times.requires_grad_()
    
        

    def refine_goal(self, last_pt):

        goal = torch.tensor([self.end_state[0], self.end_state[3], self.end_state[6]])
        if self.is_in_polyhedron(self.hpolys[-1], goal) is False:
          print("!!!!!!!!!!!!!! goal is not inside")
          for i in range(self.res-1, 0, -1):
            lba = i * 1.0 / (self.res * 1.0) 

            new_goal = (1- lba) * last_pt +  lba * goal
            
            if self.is_in_polyhedron(self.hpolys[-1], new_goal) is True:
              
              print(" new_goal",  new_goal)
              self.end_state[0] = new_goal[0]
              self.end_state[3] = new_goal[1]
              self.end_state[6] = new_goal[2]

              break
        return


    def is_in_polyhedron(self, poly, point):

        A = poly[:, 0:3]
        b = poly[:, 3]

        for i in range(len(b)):

            # print(A)
            # print(point.dtype)
            # print(b[i])
            if A[i, :].dot(point) - b[i] > 0.01:
                return False

        return True

    # just copy the functions here for convenience
    def get_inner_pts(self):

        n = self.seg
        if n <= 1:
            print("no need for pts")
            return []

        if n == 2:
            start = torch.tensor([self.start_state[0], self.start_state[3], self.start_state[6]])
            #self.refine_goal(start)
            goal  = torch.tensor([self.end_state[0], self.end_state[3], self.end_state[6]])
            pt = 0.5 * (start  + goal)
            return torch.tensor(np.array(pt))

        inner_pts = []
        for i in range(n-1):
            total_constraints = torch.vstack((self.hpolys[i], self.hpolys[i+1]))
            pt = self.get_inner_points(total_constraints.detach().numpy(), 0.01)
            #print("pt is", pt)
            if pt is None:
                print("Not valid, skip this try")
                return None
            inner_pts.append(pt)
        #print(inner_pts)
        #self.refine_goal(torch.tensor(inner_pts[-1]))
        return torch.tensor(np.array(inner_pts))
    


    def get_inner_points(self, hpoly, eps=0.001):

        A = hpoly[:, 0:3]
        b = hpoly[:, 3]

        c = [0, 0, 0, -1]
        A = np.hstack((A, np.ones(len(b)).reshape(len(b), 1)))

        #print(A)

        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        minmaxsd = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        #print(minmaxsd )
        if minmaxsd.fun is None:
            return None
        return minmaxsd.x[0:3]


##################################################constraints########################################################
    #p v a
    def get_t_state(self, t):

        if self.order == 4: 

            t_2 = t * t
            t_3 = t * t_2
            t_4 = t_2 * t_2
            t_5 = t_2 * t_3
            t_6 = t_3 * t_3
            t_7 = t_4 * t_3

            # self.times_A = torch.tensor([[    t_7,      t_6,     t_5,     t_4,     t_3,   t_2, t, 1], 
            #                             [7 * t_6,  6 * t_5, 5 * t_4, 4 * t_3, 3 * t_2, 2 * t, 1, 0],
            #                             [42* t_5, 30 * t_4, 20* t_3, 12* t_2, 6 * t,       2, 0, 0],
            #                             [210*t_4, 120* t_3, 60* t_2, 24* t,   6,           0, 0, 0]], dtype=float, requires_grad=True)
            tempA1 = torch.hstack((t_7,      t_6,     t_5,     t_4,     t_3,   t_2, t, torch.tensor(1.0)))
            tempA2 = torch.hstack((7 * t_6,  6 * t_5, 5 * t_4, 4 * t_3, 3 * t_2, 2 * t, torch.tensor(1.0), torch.zeros(1)))
            tempA3 = torch.hstack((42* t_5, 30 * t_4, 20* t_3, 12* t_2, 6 * t,       torch.tensor(2.0), torch.zeros(2)))
            tempA4 = torch.hstack((210*t_4, 120* t_3, 60* t_2, 24* t,   torch.tensor(6.0),  torch.zeros(3)))

            conti_A = torch.vstack((tempA1, tempA2, tempA3, tempA4))
        else:


            t_2 = t * t
            t_3 = t * t_2
            t_4 = t_2 * t_2
            t_5 = t_2 * t_3

            tempA1 = torch.hstack((t_5,     t_4,     t_3,   t_2, t, torch.tensor(1.0)))
            tempA2 = torch.hstack((5 * t_4, 4 * t_3, 3 * t_2, 2 * t, torch.tensor(1.0), torch.zeros(1)))
            tempA3 = torch.hstack((20* t_3, 12* t_2, 6 * t,       torch.tensor(2.0), torch.zeros(2)))

            conti_A = torch.vstack((tempA1, tempA2, tempA3))

        return conti_A


    def get_bound_state(self, t):

        if self.order == 4: 

            t_2 = t * t
            t_3 = t * t_2
            t_4 = t_2 * t_2
            t_5 = t_2 * t_3
            t_6 = t_3 * t_3
            t_7 = t_4 * t_3

            # self.times_A = torch.tensor([[    t_7,      t_6,     t_5,     t_4,     t_3,   t_2, t, 1], 
            #                             [7 * t_6,  6 * t_5, 5 * t_4, 4 * t_3, 3 * t_2, 2 * t, 1, 0],
            #                             [42* t_5, 30 * t_4, 20* t_3, 12* t_2, 6 * t,       2, 0, 0],
            #                             [210*t_4, 120* t_3, 60* t_2, 24* t,   6,           0, 0, 0]], dtype=float, requires_grad=True)
            tempA1 = torch.hstack((t_7,      t_6,     t_5,     t_4,     t_3,   t_2, t, torch.tensor(1.0)))
            tempA2 = torch.hstack((7 * t_6,  6 * t_5, 5 * t_4, 4 * t_3, 3 * t_2, 2 * t, torch.tensor(1.0), torch.zeros(1)))
            tempA3 = torch.hstack((42* t_5, 30 * t_4, 20* t_3, 12* t_2, 6 * t,       torch.tensor(2.0), torch.zeros(2)))

            conti_A = torch.vstack((tempA1, tempA2, tempA3))
        else:


            t_2 = t * t
            t_3 = t * t_2
            t_4 = t_2 * t_2
            t_5 = t_2 * t_3

            tempA1 = torch.hstack((t_5,     t_4,     t_3,   t_2, t, torch.tensor(1.0)))
            tempA2 = torch.hstack((5 * t_4, 4 * t_3, 3 * t_2, 2 * t, torch.tensor(1.0), torch.zeros(1)))
            tempA3 = torch.hstack((20* t_3, 12* t_2, 6 * t,       torch.tensor(2.0), torch.zeros(2)))

            conti_A = torch.vstack((tempA1, tempA2, tempA3))

        return conti_A




    def fill_eq_obj(self, times):

        row = 0
        #T4 = time.time()
        s_num = self.seg * self.dim * self.D - self.dim *self.D
        A = torch.tensor([])
        b = torch.zeros(self.eq_num, dtype=float)
        Q = torch.tensor([])

        # start and constraints
        for j in range(self.dim): 

            # start constraints
            idx = j * self.D
            #self.A[row:row + self.state_dim, idx:idx + self.D] = self.zero_A[0:3, :]


            tempA = torch.hstack((torch.zeros((self.state_dim, idx)), self.zero_A[0:3, :], 
                                  torch.zeros((self.state_dim, self.var_num- idx - self.D))))
            A = torch.cat((A, tempA))

            #print( self.start_state[j*self.dim:(j+1) * self.dim])
            b[row:row + self.state_dim] = self.start_state[j*self.dim:(j+1) * self.dim]
            row += self.state_dim
            
            # end constraints
            #col_start = self.var_num - (self.dim - j)*self.D

            col_start = s_num + idx


            #self.A[row:row + self.state_dim, col_start : col_start +  self.D] = self.get_t_state(times[self.seg-1])
            #print(self.end_state[j*self.dim:(j+1) * self.dim])
            b[row:row + self.state_dim] = self.end_state[j*self.dim:(j+1) * self.dim]


            tempA = torch.hstack((torch.zeros((self.state_dim, col_start)), 
                                  self.get_bound_state(times[self.seg-1]), 
                                  torch.zeros((self.state_dim, self.var_num- col_start - self.D))))
            A = torch.cat((A, tempA))
            
            row += self.state_dim
            

        for i in range(self.seg-1):
        # enforce p, dp, ddp
            start_idx = i * self.dim * self.D

            for j in range(self.dim):


                # rightA = torch.zeros([1,  self.D])
                # rightA[0, -1] = 1
        
                col_idx = start_idx + j * self.D
                next_col_idx = col_idx  + self.dim * self.D
                
                # intermediate points
                # self.A[row:row+1,  next_col_idx:next_col_idx + self.D] = rightA
                # self.b[row]   = self.inner_pts[i, j]
                # row += 1

                #continuity
                # self.A[row:row+self.order,  col_idx:col_idx + self.D] = conti_A
                # self.A[row:row+self.order,  next_col_idx:next_col_idx + self.D] = - self.zero_A

                tempA = torch.hstack((torch.zeros((self.order, col_idx)), 
                                      self.get_t_state(times[i]),
                                      torch.zeros((self.order, next_col_idx - col_idx - self.D)),
                                      - self.zero_A[0:self.order, :],
                                      torch.zeros((self.order, self.var_num - next_col_idx - self.D))))
                
                A = torch.cat((A, tempA))
    
                row += self.order

        for i in range(self.seg):

            start_idx = i * self.dim * self.D

            t = times[i].requires_grad_()
            t_2 = t * t
            t_3 = t * t_2
            t_4 = t_2 * t_2
            t_5 = t_2 * t_3


            if self.order == 3:
                m_11 = 720 * t_5
                m_12 = 360 * t_4
                m_13 = 120 * t_3

                m_22 = 192 * t_3
                m_23 = 72  * t_2

                tempCost1 = torch.hstack((m_11,  m_12, m_13))
                tempCost2 = torch.hstack((m_12,  m_22, m_23))
                tempCost3 = torch.hstack((m_13,  m_23, 36*t))

                CostQ = torch.vstack((tempCost1, tempCost2, tempCost3))

            else:
                t_6 = t_3 * t_3
                t_7 = t_4 * t_3
            

                m_11 = 100800 * t_7
                m_12 = 50400  * t_6
                m_13 = 20160  * t_5
                m_14 = 5040   * t_4

                m_22 = 25920  * t_5
                m_23 = 10800  * t_4
                m_24 = 2880   * t_3

                m_33 = 4800 * t_3
                m_34 = 1400 * t_2
                # print(times[i].requires_grad)
                # print(t.requires_grad)
                # CostQ = torch.tensor([[m_11,  m_12, m_13, m_14],
                #                         [m_12,  m_22, m_23, m_24],
                #                         [m_13,  m_23, m_33, m_34],
                #                         [m_14,  m_24, m_34, 576*t]], dtype=float)
                tempCost1 = torch.hstack((m_11,  m_12, m_13, m_14))
                tempCost2 = torch.hstack((m_12,  m_22, m_23, m_24))
                tempCost3 = torch.hstack((m_13,  m_23, m_33, m_34))
                tempCost4 = torch.hstack((m_14,  m_24, m_34, 576*t))

                CostQ = torch.vstack((tempCost1, tempCost2, tempCost3, tempCost4))
            #print(CostQ.requires_grad)
            # def print_grad(grad):
            #     print("CostQ  grad is", grad)
            # CostQ.register_hook(print_grad)
            # print(CostQ)

            for j in range(self.dim):

                col_idx = start_idx + j * self.D

                ##### objective
                # self.Q[col_idx:col_idx+self.order, col_idx:col_idx+self.order] = CostQ
        


                tempQ = torch.hstack((torch.zeros((self.order, col_idx)), 
                                      CostQ, 
                                      torch.zeros((self.order, self.var_num - col_idx - self.order))))
                
         
                Q = torch.cat((Q, tempQ, torch.zeros((self.order, self.var_num))))


           

        return Q, A, b

        

    def fill_ineq(self, times): 
        row1 = 0
        row2 = 0

        dyn_limits = torch.zeros(4, dtype=float)
        dyn_limits[0] = dyn_limits[2] = self.phy_limits[0]
        dyn_limits[1] = dyn_limits[3] = self.phy_limits[1]
    

        G1 = torch.tensor([])
        G2 = torch.tensor([])
        Q = torch.tensor([])

        h1 = torch.zeros((self.ineq_num1), dtype=float)
        h2 = torch.zeros((self.ineq_num2), dtype=float)

        #torch.hstack((self.phy_limits[0:2], self.phy_limits[0:2]))
        # 1. dynamic constraints
        for i in range(self.seg):
        # enforce p, dp, ddp
            step_t = times[i] / self.res
            start_idx = i * self.dim * self.D
            const_num = self.hpolys[i].shape[0]
            #print("=========== step ", step_t)

            for step in range(self.res):
                #print(step)

                #T4 = time.time()
                cur_t = step * step_t
                #print("t is ", cur_t)
                #dynamic constraints
                if cur_t == 0:
                    G = self.zero_A
                else:
                    G = self.get_t_state(cur_t)  # p, v, a
                
               
                # T3 = time.time()
                # print('ineuqlity Time is :%s ms' % ((T3 - T4)*1000))
                # self.G1[row1:row1+const_num, col_idx:col_idx + self.D] = self.hpolys[i][:, j].reshape(const_num, 1)@G[0, :].reshape(1, self.D)
                
                tempG1  = torch.hstack((torch.zeros((const_num, start_idx)), 
                                        self.hpolys[i][:, 0].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        self.hpolys[i][:, 1].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        self.hpolys[i][:, 2].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        torch.zeros((const_num, self.var_num - start_idx  - 3 *  self.D))))
            
                
            
                G1 = torch.cat((G1, tempG1))

                h1[row1:row1+const_num] = self.hpolys[i][:, 3]
                row1 += const_num


                dynG =  torch.vstack((G[1:3, :], -G[1:3, :]))
                for j in range(self.dim):

                    col_idx = start_idx + j * self.D
                    # print(dynG.shape)
                    #self.G2[row2:row2+4, col_idx:col_idx + self.D] = dynG

                    tempG2 = torch.hstack((torch.zeros((4, col_idx)), 
                                          dynG, 
                                          torch.zeros((4, self.var_num - col_idx - self.D))))
            
            
            
                    G2 = torch.cat((G2, tempG2))
            

                
                    h2[row2:row2+4] = dyn_limits
                    row2 += 4


        return G1, h1, G2, h2 
    


    def fill_phase1_ineq(self, times): 
        row1 = 0
        row2 = 0



        G1 = torch.tensor([])
        G2 = torch.tensor([])
        Q = torch.tensor([])

        h1 = torch.zeros((self.ineq_num1), dtype=float)
        h2 = torch.zeros((self.ineq_num2), dtype=float)


        dyn_limits = torch.zeros(4, dtype=float)
        dyn_limits[0] = dyn_limits[2] = self.phase1_phy_limits[0]
        dyn_limits[1] = dyn_limits[3] = self.phase1_phy_limits[1]
        #print(dyn_limits.dtype)
        #torch.hstack((self.phy_limits[0:2], self.phy_limits[0:2]))
        # 1. dynamic constraints
        for i in range(self.seg):
        # enforce p, dp, ddp
            step_t = times[i] / self.res
            start_idx = i * self.dim * self.D
            const_num = self.hpolys[i].shape[0]
            #print("=========== step ", step_t)

            for step in range(self.res):

                #T4 = time.time()
                cur_t = step * step_t
                #print("t is ", cur_t)
                #dynamic constraints
                if cur_t == 0:
                    G = self.zero_A
                else:
                    G = self.get_t_state(cur_t)  # p, v, a
               
                # T3 = time.time()
                # print('ineuqlity Time is :%s ms' % ((T3 - T4)*1000))
                #print(G.dtype)
                
                tempG1  = torch.hstack((torch.zeros((const_num, start_idx)), 
                                        self.hpolys[i][:, 0].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        self.hpolys[i][:, 1].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        self.hpolys[i][:, 2].reshape(const_num, 1)@G[0, :].reshape(1, self.D),
                                        torch.zeros((const_num, self.var_num - start_idx  - 3 *  self.D))))
            
                
            
                G1 = torch.cat((G1, tempG1))
            
                # for j in range(self.dim):

                #     col_idx = start_idx + j * self.D
                #     self.G1[row1:row1+const_num, col_idx:col_idx + self.D] = self.hpolys[i][:, j].reshape(const_num, 1)@G[0, :].reshape(1, self.D)
                
                h1[row1:row1+const_num] = self.hpolys[i][:, 3]
                row1 += const_num


                dynG =  torch.vstack((G[1:3, :], -G[1:3, :]))
                for j in range(self.dim):

                    col_idx = start_idx + j * self.D
                    # print(dynG.shape)
                    #self.G2[row2:row2+4, col_idx:col_idx + self.D] = dynG
                
                    tempG2 = torch.hstack((torch.zeros((4, col_idx)), 
                                          dynG, 
                                          torch.zeros((4, self.var_num - col_idx - self.D))))
            
            
            
                    G2 = torch.cat((G2, tempG2))
            

                    h2[row2:row2+4] = dyn_limits
                    row2 += 4
    
    
        return G1, h1, G2, h2

    def solve(self):

        P = sparse.csc_matrix(self.Q.detach().numpy())
        q = np.zeros(self.var_num)
        A = sparse.vstack([self.A.detach().numpy(), self.G1.detach().numpy(), self.G2.detach().numpy()], format='csc')
        l = np.hstack([self.b.detach().numpy(),  -math.inf * np.ones(self.ineq_num)])
        u = np.hstack([self.b.detach().numpy(),  self.h1.detach().numpy(), self.h2.detach().numpy()])

        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)


        res = prob.solve()

        #print(res)
        if res.info.status != 'solved':
            print('OSQP did not solve the problem!')
            return

        self.coeffs = []
        # step one recover coeffs
        for i in range(self.seg):
            coeff = torch.zeros((self.dim, self.D)) 

            for j in range(self.dim):
                for k in range(self.D):
                    idx         = i * self.dim * self.D + j * self.D + k
                    coeff[j, k] = res.x[idx]
            
            #print(coeff)

            self.coeffs.append(coeff.detach().numpy())

        # print(self.T.detach().numpy())
        # print("self.seg  ", self.seg)

        self.Traj = Trajectory(self.coeffs, self.T.detach().numpy())

        

    def update_traj(self, res):

        self.coeffs = []
        # step one recover coeffs
        for i in range(self.seg):
            coeff = torch.zeros((self.dim, self.D)) 

            for j in range(self.dim):
                for k in range(self.D):
                    idx         = i * self.dim * self.D + j * self.D + k
                    coeff[j, k] = res[idx]
            
            #print(coeff)

            self.coeffs.append(coeff.detach().numpy())
        
        print(self.Times[0:self.seg])
        self.Traj = Trajectory(self.coeffs, self.Times[0:self.seg].detach().numpy())




    def vis_traj(self, res = 0.1, i = 0):


        import matplotlib.pyplot as plt
        import numpy as np
        
        if self.Traj is None:
            return

        t = np.arange(0, self.Traj.dur, res)


        # setting the corresponding y - coordinates
        pts  = self.Traj.get_sample_pts(res)
        vels = self.Traj.get_sample_vels(res)
        accs = self.Traj.get_sample_accs(res)
        
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(t, pts[:, 0], label='pos_x')
        plt.plot(t, pts[:, 1], label='pos_y')
        plt.plot(t, pts[:, 2], label='pos_z')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, vels[:, 0], label='vel_x')
        plt.plot(t, vels[:, 1], label='vel_y')
        plt.plot(t, vels[:, 2], label='vel_z')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, accs[:, 0], label='acc_x')
        plt.plot(t, accs[:, 1], label='acc_y')
        plt.plot(t, accs[:, 2], label='acc_z')

        plt.legend()
        #plt.show()
        plt.savefig(f'min_traj{i}.png')


    def vis_sfc_with_trajs(self, res = 0.1,  i = 0):

        import plotly.graph_objects as go
        fig = go.Figure()
        
        import numpy as np
        
        if self.Traj is None:
            return

        t = np.arange(0, self.Traj.dur, res)


        # setting the corresponding y - coordinates
        pts  = self.Traj.get_sample_pts(res)

        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='lines'))
       

        for hpoly in self.hpolys:

            all_const = hpoly.detach().numpy()
            print(all_const)
            inner = self.get_inner_points(all_const)
 
            all_const[:, 3] = - all_const[:, 3]

            halfspaces = scipy.spatial.HalfspaceIntersection(all_const, inner)
            points = halfspaces.intersections

            #print(halfspaces.intersections)
            # only for vis
            for simplex in scipy.spatial.ConvexHull(points).simplices:

                #print(simplex )
                vect = points[simplex, :]
                fig.add_trace(go.Mesh3d(x=points[simplex, 0], y=points[simplex, 1],  z=points[simplex, 2],  color="orange", opacity=.1))
                
        fig.write_image(f'corridor{i}.png')



if __name__ == '__main__':


    from corridor_generator import CorridorGenerator
    import open3d as o3d
    import numpy as np
    cloud = o3d.io.read_point_cloud(
        'datasets/single_map_dataset/map.pcd')
    if cloud.is_empty():
        exit()
    points = np.asarray(cloud.points)

    print("points.shape： ", points.shape)

    start = np.array([-2.27254446, -0.6393239, 1.22098313])
    goal = np.array([0.35536006, -10.43656709, 0.17809835])
    map_range = ([-10, -10, -0.1], [10, 10, 3])
    safe_distance =0.5

    sfc_generator = CorridorGenerator(start, goal, map_range, safe_distance, points, 5, 5.0)
    sfc_generator.get_corridor()
    np.set_printoptions(threshold=np.inf)


    hpolys = sfc_generator.getHpolys()


    dim = 3

    start_state = torch.zeros((dim, 3))# p, v, a  
    # px, vx, ax, 
    # py, vy, ay, 
    # pz, vz, az.
    #position
    start_state[0, :] = start
    #velocity
    start_state[1, :] = [0.5, 0.0, 0.0]
    start_state[2, :] = [0.1, 0.0, 0.0]
    end_state = torch.zeros((dim, 3))
    end_state[:, 0] = goal


    state = np.concatenate((start_state, end_state), axis=None)

    print(state)

    qp_traj = MinTrajOpt(state, [], hpolys, params)
    status = qp_traj.solve()
    qp_traj.vis_traj(i =1)

    t = 1.1 * qp_traj.T


    qp_traj.resetT(t)
    status = qp_traj.solve()
    qp_traj.vis_traj(i =2)

