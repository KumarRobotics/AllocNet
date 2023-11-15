from utils.rrt3D import rrt
import math
import scipy
import numpy as np
import os
import time
import irispy
import sys
sys.path.append("..")


class CorridorGenerator:

    def __init__(self, start, goal, map_range, safe_dis, obs_points, max_num, planner_timeout_threshold):

        self.start = start
        self.goal = goal
        self.map_range = map_range
        self.safe_dis = safe_dis

        self.obs_points = obs_points
        self.max_num = max_num

        self.planner_timeout_threshold = planner_timeout_threshold

        # results
        self.polyhedrons = []
        self.path = []
        self.idices = []
        self.past_q_pts = []

    def iris(self, query_point, dis_poly=False):

        #print("query_point", query_point)

        stuck_thresh = 5

        # Append the last 5 query points to the list
        # if len(self.past_q_pts) < stuck_thresh:
        #     self.past_q_pts.append(query_point)
        # else:
        #     if (np.equal(query_point, self.past_q_pts[0]) & np.equal(query_point, self.past_q_pts[1]) & np.equal(query_point, self.past_q_pts[2]) & np.equal(query_point, self.past_q_pts[3]) & np.equal(query_point, self.past_q_pts[4])).all():
        #         print("stuck at the same point for 5 times!")
        #         return None
        #     self.past_q_pts.pop(0)
        #     self.past_q_pts.append(query_point)

        right = query_point + [3, 3, 2]
        left = query_point - [3, 3, 2]
        bounds = irispy.Polyhedron.from_bounds(left, right)
        region, debug = irispy.inflate_region(self.obs_points, query_point, bounds=bounds, return_debug_data=True)
        return region.polyhedron

    # main interface

    def get_corridor(self):
        # print("start to get corridor")
        hpoly_elems = None
        hpoly_end_probs = []
        hpoly_seq_end_probs = []

        vpoly_elems = None
        vpoly_end_probs = []
        vpoly_seq_end_probs = []


        # step one, path search
        planner = rrt(self.start, self.goal, self.map_range, self.safe_dis, self.obs_points)

        # If the execution time of the planner exceeds the threshold, we stop the planner and return None
        self.path = planner.run(self.planner_timeout_threshold)

        if self.path is None:
            print("path is None")
            return None

        # step two, get convex cover
        self.polyhedrons.clear()
        n = len(self.path)
        print("[Corridor Generation] : path length is: ", n)
        #print("self.max_num is: ", self.max_num)
        #print(self.path)
        res = 10
        progress = res * math.ceil(n * 1.0 / ((self.max_num) * 1.0))
        #print("progress ", progress)
        i = 0
        progress_cnt = 0
        #print(self.path)

        # discrete the path
        temp_path = []

        for i in range(n-1):
            for j in range(res):
                new_pt = self.path[i] + (j * 1.0 / res * 1.0) * \
                    (self.path[i+1] - self.path[i])
                temp_path.append(new_pt)
        cor_num = 0

        while (i < len(temp_path)):
            #print("enter while loop")
            q_pt = temp_path[i]
            #print(q_pt)

            if len(self.polyhedrons) == 0:
                # print(q_pt)
                cur_poly = self.iris(q_pt)

                if cur_poly is None:
                    return None
                #print("corridor num ", cor_num)
                self.polyhedrons.append(cur_poly)
                cor_num += 1

                i += 1
            else:
                if self.is_in_polyhedron(cur_poly, q_pt):

                    if progress_cnt < progress:
                        i += 1
                        progress_cnt += 1
                        #print("stay, ", i)
                        continue

                if self.is_in_polyhedron(cur_poly, q_pt) == False:
                    new_p_t = temp_path[i-1]
                else:
                    new_p_t = q_pt
                
                i += 1
                    
                cur_poly = self.iris(new_p_t)
                progress_cnt = 0

                # if cur_poly is None:
                #     return None
                    
                overlap = self.overlap(self.polyhedrons[-1], cur_poly, new_p_t, 0.01)

                if overlap == False:
                    print("The path is not fully collision-free, abort this try")
                    return None
                # print("self.is_in_polyhedron(cur_poly, q_pt)   ", self.is_in_polyhedron(cur_poly, new_p_t))

                # print("self.is_in_polyhedron(cur_poly, temp_path[i])   ", self.is_in_polyhedron(cur_poly, temp_path[i]))

                # print("new_p_t   ", new_p_t)
                # print("temp_path[i]   ", temp_path[i])


                #print("corridor num ", cor_num)
                self.polyhedrons.append(cur_poly)
                cor_num += 1

        # print(temp_path)
        #if len(self.polyhedrons) > self.max_num:
        self.short_cut_corridor()
        
        if len(self.polyhedrons) <= 0:
            return None

        print("[Corridor Generation] =========================== corridor len is: ", len(self.polyhedrons))
        for cur_poly in self.polyhedrons:
            hpoly = self.getHpoly(cur_poly)
            vpoly = np.asarray(cur_poly.generatorPoints())

            # print("hpoly: ", hpoly)
            # print("shape of hpoly: ", hpoly.shape)
            # print("vpolys: ", vpoly)
            # print("shape of vpoly: ", vpoly.shape)

            if hpoly_elems is None:
                hpoly_elems = hpoly
            else:
                hpoly_elems = np.vstack((hpoly_elems, hpoly))

            if vpoly_elems is None:
                vpoly_elems = vpoly
            else:
                vpoly_elems = np.vstack((vpoly_elems, vpoly))

            cur_hpoly_row = hpoly.shape[0]
            hpoly_end_probs.extend([0] * (cur_hpoly_row - 1) + [1])
            hpoly_seq_end_probs.extend([0] * cur_hpoly_row)

            cur_vpoly_row = vpoly.shape[0]
            vpoly_end_probs.extend([0] * (cur_vpoly_row - 1) + [1])
            vpoly_seq_end_probs.extend([0] * cur_vpoly_row)

        hpoly_seq_end_probs[-1] = 1
        hpoly_end_probs = np.asarray(hpoly_end_probs)
        hpoly_seq_end_probs = np.asarray(hpoly_seq_end_probs)

        vpoly_seq_end_probs[-1] = 1
        vpoly_end_probs = np.asarray(vpoly_end_probs)
        vpoly_seq_end_probs = np.asarray(vpoly_seq_end_probs)

        return hpoly_elems, hpoly_end_probs, hpoly_seq_end_probs, vpoly_elems, vpoly_end_probs, vpoly_seq_end_probs



    def getHpolys(self):

        hpolys = []

        for poly in self.polyhedrons: 

            A = poly.getA()
            b = poly.getB().reshape((-1, 1))

            poly = np.hstack((A, b))
            hpolys.append(poly)

        return hpolys


    def getHpoly(self, poly):

        A = poly.getA()
        b = poly.getB().reshape((-1, 1))

        hpoly = np.hstack((A, b))

        return hpoly


    def getVpolys(self):

        vpolys = []

        for poly in self.polyhedrons: 

            convex_points = np.asarray(poly.generatorPoints())

            vpolys.append(convex_points)

        return vpolys


    def getVpoly(self, poly):
       
        convex_points = np.asarray(poly.generatorPoints())

        return convex_points




    def short_cut_corridor(self):

        temp_corridor = self.polyhedrons.copy()

        M = len(temp_corridor)
        if len(temp_corridor) == 1:
            temp_corridor.append(temp_corridor.front)
            return

        overlap = False
        self.idices = []
        self.idices.append(M - 1)

        for i in range(M-1, -1, -1):
            #print("i is", i)
            for j in range(0, i):
                if j < i-1:
                    overlap = self.overlap(temp_corridor[i], temp_corridor[j], 0.01)
                else:
                    overlap = True

                if overlap:
                    if j not in self.idices:
                        self.idices.insert(0, j)
                    i = j+1
                    break
        #print(self.idices)

        self.polyhedrons.clear()

        for ele in self.idices:
            self.polyhedrons.append(temp_corridor[ele])

        if len(self.polyhedrons) > self.max_num:
            self.polyhedrons = self.polyhedrons[0:self.max_num]
            self.idices = self.idices[0:self.max_num]



    def overlap(self, poly1, poly2, eps=0.001):

        total_constraints = np.vstack((self.getHpoly(poly1), self.getHpoly(poly2)))
        
        A = total_constraints[:, 0:3]
        b = total_constraints[:, 3]
        c = np.asarray([0.0, 0.0, 0.0, -1.0])
        A = np.hstack((A, np.ones(len(b)).reshape(len(b), 1)))
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        minmaxsd = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        #print(minmaxsd)
        if minmaxsd.fun is None:
            return False
        return minmaxsd.fun < -eps
    


    def overlap(self, poly1, poly2, x0, eps=0.001):
        total_constraints = np.vstack((self.getHpoly(poly1), self.getHpoly(poly2)))
        
        A = total_constraints[:, 0:3]
        b = total_constraints[:, 3]
        c = np.asarray([0.0, 0.0, 0.0, -1.0])
        A = np.hstack((A, np.ones(len(b)).reshape(len(b), 1)))
        #x0 =  np.hstack((x0, 0.0001))
        # print(x0)
        # print(b - A@x0)
        # print(c.dot(x0))
        #self.two_corridors_inside(poly1, poly2)

        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        minmaxsd = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        #print(minmaxsd)
        if minmaxsd.fun is None:
            return False
        return minmaxsd.fun < -eps

    

    def two_corridors_inside(self, poly1, poly2):


        points1 = self.getVpoly(poly1)

        flag1 = True
        for pt in points1:
            if self.is_in_polyhedron(poly2, pt) == False:
                flag1 = False
                break

        points2 = self.getVpoly(poly2)

        flag2 = True
        for pt in points2:
            if self.is_in_polyhedron(poly1, pt) == False:
                flag2 = False
                break
        #print("flag 1 is", flag1)
        #print("flag 2 is", flag2)
        return flag2 or flag1  

    def get_overlap_points(self, poly1, poly2, eps=0.001):
        total_constraints = np.vstack((self.getHpoly(poly1), self.getHpoly(poly2)))
        #print(total_constraints)
        A = total_constraints[:, 0:3]
        b = total_constraints[:, 3]
        c = [0, 0, 0, -1]
        A = np.hstack((A, np.ones(len(b)).reshape(len(b), 1)))

        # print(A)
        # print(b)
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        minmaxsd = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds)
        #print(minmaxsd)
        if minmaxsd.fun is None:
            return None
        return minmaxsd.x[0:3]

    def is_in_polyhedron(self, poly, point):

        hpoly = self.getHpoly(poly)
        
        A = hpoly[:, 0:3]
        b = hpoly[:, 3]

        for i in range(len(b)):

            # print(A[i, :])
            # print(point)
            # print(b[i])
            if A[i, :].dot(point) + 1e-4 > b[i]:
                return False

        return True

    def get_inner_pts(self):

        n = len(self.polyhedrons)
        if n <= 0:
            print("No corridor no path!")
            return None, self.goal

        if n == 1:
            pt = 0.5 * (self.goal + self.start)
            return np.array([pt]), self.goal


        inner_pts = []
        for i in range(n-1):

            pt = self.get_overlap_points(self.polyhedrons[i], self.polyhedrons[i+1], 0.01)
            if pt is None:
                print("Not valid, skip this try")
                return None, self.goal

            inner_pts.append(pt)
        
        new_goal = self.refine_goal(inner_pts[-1])

        # print( np.asarray(inner_pts))
        return np.asarray(inner_pts), new_goal


    def refine_goal(self, last_pt):

        res = 10

        if self.is_in_polyhedron(self.polyhedrons[-1], self.goal) is False:
          print("!!!!!!!!!!!!!! goal is not inside")
          for i in range(10, 0, -1):
            lba = i * 1.0 / (res* 1.0) 
            new_goal = (1- lba) * last_pt +  lba * self.goal
            
            if self.is_in_polyhedron(self.polyhedrons[-1], new_goal) is True:

              print(" new_goal is inside the corridor,  the lba is , ",  new_goal, lba)
                      
              return new_goal
          return  (1- 0.005) * last_pt +  0.005 * self.goal
        

        return self.goal



    def vis_obs_points(self):
        import plotly.graph_objects as go

        print(self.obs_points.shape)
        fig = go.Figure(
            data=[go.Scatter3d(
                x=self.obs_points[:, 0], y=self.obs_points[:,
                                                           1], z=self.obs_points[:, 2],
                mode='markers',
                marker=dict(size=1, color="blue"))],
            layout=dict(scene=dict(xaxis=dict(visible=False), yaxis=dict(
                visible=False), zaxis=dict(visible=False)))
        )
        fig.show()

    def vis_corridor(self, ObsPts=True):
        import plotly.graph_objects as go

        plot_path = np.array(self.path)

        if self.path is None:
            return

        if ObsPts:
            print("print obs")
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=self.obs_points[:, 0], y=self.obs_points[:,
                                                               1], z=self.obs_points[:, 2],
                    mode='markers',
                    marker=dict(size=1, color="blue"))])

        else:
            fig = go.Figure()

        plot_path = np.array(self.path)
        fig.add_trace(go.Scatter3d(
            x=plot_path[:, 0], y=plot_path[:, 1], z=plot_path[:, 2], mode='lines'))
       
        all_poly_points = self.getVpolys()


        for convex_points in all_poly_points:

            #print(convex_points)
            
            # only for vis
            for simplex in scipy.spatial.ConvexHull(convex_points).simplices:
                fig.add_trace(go.Mesh3d(
                    x=convex_points[simplex, 0], y=convex_points[simplex, 1],  z=convex_points[simplex, 2],  color="orange", opacity=.1))
        fig.show()


if __name__ == '__main__':

    import open3d as o3d
    import numpy as np
    cloud = o3d.io.read_point_cloud(
        'datasets/single_map_dataset/map.pcd')
    if cloud.is_empty():
        exit()
    points = np.asarray(cloud.points)

    # print(cloud)
    print("points.shape[0]: ", points.shape[0])
    # print("points: ", points)

    start = np.array([-2.27254446, -0.6393239, 1.22098313])
    goal = np.array([0.35536006, -10.43656709, 0.17809835])

    # is the search range, could be smaller than the map size to save computation
    map_range = ([-10, -10, -0.1], [10, 10, 3])
    safe_distance =0.5

    # you can restrain the num of polytope here


    for i in range(50):

        rand_start_x = np.random.uniform(-10,10)
        rand_start_y = np.random.uniform(-10,10)
        rand_start_z = np.random.uniform(-0.1, 3)

        rand_end_x = np.random.uniform(-10, 10)
        rand_end_y = np.random.uniform(-10, 10)
        rand_end_z = np.random.uniform(-0.1, 3)

        start = np.array([rand_start_x, rand_start_y, rand_start_z])
        goal = np.array([rand_end_x, rand_end_y, rand_end_z])

        print("start: ", start)
        print("goal: ", goal)

        sfc_generator = CorridorGenerator( start, goal, map_range, safe_distance, points, 5, 5.0)
        sfc_generator.get_corridor()
        inner = sfc_generator.get_inner_pts()
