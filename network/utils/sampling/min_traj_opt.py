import numpy as np
import scipy
import cvxpy as cp
from trajectory import Trajectory

class MinTrajOpt:

    def __init__(self, start_state, goal, T, inner_pts, order = 3):
        

        self.T = T
        print("T is", T)
        
        self.inner_pts = inner_pts
        a = np.array([[1, 2, 1], [3, 1, 2]])
        print("a is", a)
        print("inner_pts is", inner_pts)
        self.order = order
        self.seg = len(T)
        print("seg is", self.seg)
        self.dim = len(goal)
        self.state_dim = start_state.shape[1]
        print("self.state_dim is", self.state_dim)

        #print("dim is", self.dim)
        self.start_state = start_state
        self.end_state = np.zeros((self.dim, 3))
        self.end_state[:, 0] = goal

        print("self.start_state is", self.start_state)
        print("self.end_state is", self.end_state)



        print("self.end_state is", self.end_state)
        degree = 2 * order - 1
        self.D = degree + 1

        print("self.D is", self.D)
        print("self.order is", self.order)
        print("self.dim is", self.dim)
        print("self.seg is", self.seg)
        self.var_num = self.seg * self.dim * self.D
        
        print("self.var_num  is", self.var_num)

        self.total_const_num = (2 * self.state_dim + ( 2 + self.order ) * (self.seg-1) ) * self.dim 
        print("total_const_num  is", self.total_const_num)
        self.A = np.zeros((self.total_const_num, self.var_num))
        self.b = np.zeros(self.total_const_num)

        self.Csqrt = np.zeros((self.var_num, self.var_num))

        self.fill_equality()

        # results
        self.Coeffs = np.zeros((1, self.var_num)) 
        self.Traj = None

    # one continuity constraints    p_i(t_i) = p_{i+1}(0)
    def get_continuity(self, t):

        leftA = np.zeros([self.order,  self.D])
        temp = list(range(self.D-1, -1, -1))

        # 7 6 5 4 3 2 1
        s_order = 2
        for i in range(self.order):
            for j in range(self.D):
                #print(j)
                order = self.D - s_order - j
                leftA[i, j] = temp[j] * pow(t, order)
                temp[j] = temp[j] * order
            
            s_order += 1
            
        #print(leftA)  
        rightA = np.zeros([self.order,  self.D])
        n = 2
        m = 1
        for i in range(self.order):
            rightA[i, self.D-2-i] = -m
            m *= n
            n += 1
            
        #print(rightA)
        return leftA, rightA

    #ctrl = 3, p, v, a constraints
    def get_intermediate_pos(self, t, p_j, ctrl=3):

        #left point + right point
        right = np.zeros([2,  self.D])
        right[1, -1] = 1

        
        left = np.zeros([2,  self.D])
        for i in range(self.D):
            left[0, i] = pow(t, self.D-1-i)

        return left, right


    def get_start_pos(self):

        A = np.zeros([self.state_dim,  self.D])
        n = 1
        m = 1
        for i in range(self.state_dim):
            A[i, self.D-1-i] = m
            m *= n
            n += 1
            
        #print(A)
        return A
        

    def get_end_pos(self, t):

        A = np.zeros([self.state_dim,  self.D])

        temp = np.ones(self.D)

        # 7 6 5 4 3 2 1
        s_order = 1
        for i in range(self.state_dim):
            for j in range(self.D):
                #print(j)
                order = self.D - s_order - j
                A[i, j] = temp[j] * pow(t, order)
                temp[j] = temp[j] * order
            
            s_order += 1
        
        #print(A)    
        return A
        
   
    def get_minimal_control(self,t):

        if self.order == 3:
            csqrt = scipy.linalg.sqrtm(np.array([[720*pow(t, 5),   360*pow(t, 4), 120*pow(t, 3)],
                                                [360*pow(t, 4),   192*pow(t, 3), 72*pow(t, 2)],
                                                [120*pow(t, 3),    72*pow(t, 2), 36*t]]))
        elif self.order == 4:
            csqrt = scipy.linalg.sqrtm(np.array([[100800*pow(t, 7), 50400*pow(t, 6), 20160*pow(t, 5), 5040*pow(t, 4)],
                                                 [50400*pow(t, 6),  25920*pow(t, 5), 10800*pow(t, 4), 2880*pow(t, 3)],
                                                 [20160*pow(t, 5),  10800*pow(t, 4),  4800*pow(t, 3), 1440*pow(t, 2)],
                                                 [5040*pow(t, 5),    2880*pow(t, 4),  1440*pow(t, 3),  576*t],]))
        
        return csqrt






    def fill_equality(self):

        row = 0
        # start constraints
        for j in range(self.dim): 
            #print(self.T[self.seg-1])
            #print(self.A[row:row + self.state_dim, j * self.dim : j * self.dim + self.D ])
            idx = j * self.D
                
            self.A[row:row + self.state_dim, idx:idx + self.D] = self.get_start_pos()
            print("dddddddddddddd", self.start_state[j, :])
            self.b[row:row + self.state_dim] = self.start_state[j, :]
            

            row += self.state_dim
            
        # end constraints
        for j in range(self.dim): 
            temp = self.get_end_pos(self.T[self.seg-1])
            #print(temp.shape)
            temp2 = self.A[row:row+self.state_dim, 0:self.D]
            #print(self.T[self.seg-1])
            col_start = self.var_num - self.dim*self.D   + j * self.D
            self.A[row:row+self.state_dim, col_start : col_start +  self.D] = self.get_end_pos(self.T[self.seg-1])
            #print(self.end_state[j, :])
            self.b[row:row+self.state_dim] = self.end_state[j, :]
            #print("self.A[row:row+self.state_dim] is", self.A[row:row+self.state_dim])
            #print("self.b[row:row+self.state_dim] is", self.b[row:row+self.state_dim])
            row += self.state_dim
            
        
        for i in range(self.seg-1):
        # enforce p, dp, ddp
            start_idx = i * self.dim * self.D
            for j in range(self.dim):

                leftA, rightA = self.get_intermediate_pos(self.T[i], self.inner_pts[i, j])

                col_idx = start_idx + j * self.D
                next_col_idx = col_idx  + self.dim * self.D

                self.A[row:row+2,  col_idx:col_idx + self.D] = leftA
                self.A[row:row+2,  next_col_idx:next_col_idx + self.D] = rightA
                
                self.b[row:row+2] = np.array([self.inner_pts[i, j], self.inner_pts[i, j]])
                row += 2

                leftA, rightA = self.get_continuity(self.T[i])
                self.A[row:row+self.order,  col_idx:col_idx + self.D] = leftA
                self.A[row:row+self.order,  next_col_idx:next_col_idx + self.D] = rightA
                self.b[row:row+self.order] = np.zeros(self.order)
                row += self.order


        # print("ros is ", row)
        # print(self.A)
        # print(self.b)

        for i in range(self.seg):
            t = self.T[i]
            for j in range(self.dim):
                start_idx = i * self.dim * self.D + j * self.D
                # print("start_index is ", start_idx)
                # print("t is, ", t)
                self.Csqrt[start_idx:start_idx+self.order, start_idx:start_idx+self.order] = self.get_minimal_control(t)

              

    



    def solve(self):

        z = cp.Variable(self.var_num)

        def f_(z,Csqrt,A,b):
            return  cp.sum_squares(Csqrt@z) if isinstance(z, cp.Variable) else torch.sum((Csqrt@z)**2)
        def g_(z,Csqrt,A,b):
            return G@z - h
        def h_(z,Csqrt,A,b):
            return A@z - b

        cp_equalities = [eq(z, self.Csqrt, self.A, self.b) == 0 for eq in [h_]]
        problem = cp.Problem(cp.Minimize(f_(z, self.Csqrt, self.A, self.b)), cp_equalities)
        status = problem.solve(cp.OSQP)
        print(status)
        #print("result is ... ", z.value)
        self.Coeffs = z.value

        self.get_traj()

        return status
    

    #seg * dim *D
    def get_traj(self):

        coeffs = []
        # step one recover coeffs
        # for i in range(self.seg):
        #     coeff = np.zeros((self.dim, self.D)) 

        #     for j in range(self.dim):
        #         for k in range(self.D):
        #             idx         = i * self.dim * self.D + j * self.D + k
        #             coeff[j, k] = self.Coeffs[idx]
            
        #     #print(coeff)

        for i in range(self.seg):
            coeff = np.zeros((self.dim, self.D))
            for j in range(self.dim):
                for k in range(self.D):
                    idx = j * self.D + k

                    coeff[j, k] = self.Coeffs[i * self.dim * self.D + idx]



            coeffs.append(coeff)

        
        self.Traj = Trajectory(coeffs, self.T)
    

    def vis_traj(self, res = 0.02):


        import matplotlib.pyplot as plt
        import numpy as np
        
        if self.Traj is None:
            return

        t = np.arange(0, self.Traj.dur, res)
        # setting the corresponding y - coordinates
        pts  = self.Traj.get_sample_pts(res)
        vels = self.Traj.get_sample_vels(res)
        accs = self.Traj.get_sample_accs(res)

        #print last point
        #print(pts[:, -1])
        

        plt.subplot(3, 1, 1)
        plt.plot(t, pts[:, 0], label='pos_x')
        plt.plot(t, pts[:, 1], label='pos_y')
        plt.plot(t, pts[:, 2], label='pos_z')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, vels, label='vel')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, accs, label='acc')
        plt.grid()
        plt.legend()
        plt.show()

        #vis trajectory in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label='pos')
        plt.show()

    def vis_save_and_show(self, i = 0):

        #set figure background color, axis is equal and color scale is Viridis
        # axis is black and background is white
        #set length of axis
        self.fig.update_layout(template="plotly_white",
                               scene = dict(aspectmode='data',
                                            xaxis=dict(range=[-10, 10]),
                                            yaxis=dict(range=[-10, 10]),
                                            zaxis=dict(range=[-1, 4]),
                                            xaxis_title='X',
                                            yaxis_title='Y',
                                            zaxis_title='Z'))
                             
        # add grid
        #set size of the figure
        self.fig.update_layout(width=1200, height=800)
        #save figures
        # self.fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        # self.fig.update_zaxes(scaleanchor = "x", scaleratio = 1)
        self.fig.write_image(f'corridor{i}.png')
        self.fig.show()
    

if __name__ == '__main__':

    import yaml

    
    np.set_printoptions(threshold=np.inf)

    with open("params.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            

    #dynamic feasibility
    max_vel = params['physical_limits']['max_vel']
    max_acc = params['physical_limits']['max_acc']


    import open3d as o3d
    import numpy as np
    cloud = o3d.io.read_point_cloud(
        '/home/wyw/Code/AirSim_Blocks_Modified_Linux_Build/nn_corridor_generator/nn_bak/datasets/single_map_dataset/map.pcd')
    if cloud.is_empty():
        exit()
    points = np.asarray(cloud.points)

    #print("points.shape： ", points.shape)

    #start = np.array([-2.27254446, -0.6393239, 1.22098313])
    #goal = np.array([0.35536006, -10.43656709, 0.17809835])
    goal = np.array([5, -8 , 2])
    start = np.array([-3, 5, 0.5])
    map_range = ([-10, -10, -0.1], [10, 10, 3])
    safe_distance =0.5

    #generate 10 different distinct colors
    from random import randint
    colors = []
    # for i in range(10):
    #     colors.append('#%06X' % randint(0, 0xFFFFFF))

    #use color style
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # trajectory porperties
    order = params['planning']['order']
    dim = 3
    start_state = np.zeros((dim, 3))# p, v, a  
    # px, vx, ax, 
    # py, vy, ay, 
    # pz, vz, az.
    #position
    start_state[:, 0] = start
    #velocity
    start_state[:, 1] = np.array([0, 0, 0])
    #acceleration
    start_state[:, 2] = np.array([0, 0, 0])


    num = randint(3, 10)
    T = np.ones(num)
    Inner_pts = np.zeros((num, dim))

    #generate random points
    for i in range(num):
        Inner_pts[i, :] = np.array([randint(-10, 10), randint(-10, 10), randint(0, 3)])

    qp_traj1 = MinTrajOpt(start_state, goal, T, Inner_pts, order)
    status = qp_traj1.solve()
    qp_traj1.vis_traj()
