import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os
import sys
import math

class rrt():
    def __init__(self, start, goal, bounds, safe_dis, obs_points):
        

        self.obs_points = obs_points
        self.kdtree = KDTree(obs_points)
    
        self.bounds = bounds
        self.safe_dis = safe_dis
        self.Parent = {}
        self.V = []
        # self.E = edgeset()
        self.i = 0
        self.maxiter = 5000
        self.stepsize = 1.0
        self.Path = []
        self.done = False
        self.x0 = tuple(start)
        self.xt = tuple(goal)
        
        self.ind = 0
        # self.fig = plt.figure(figsize=(10, 8))

    def clear(self):
        self.Parent = {}
        self.V = []
        # self.E = edgeset()
        self.Path = []

    def wireup(self, x, y):
        # self.E.add_edge([s, y])  # add edge
        self.Parent[x] = y

    def run(self, timeout_threshold):

 
        if self.is_inside_obs(self.x0) or self.is_inside_obs(self.xt):
            print('start or end inside!')
            return None
            

        self.V.append(self.x0)
        while self.ind < self.maxiter:
            xrand = self.sampleFree()    # sample random point which is collision-free
            if xrand is None:
                self.ind += 1
                continue
            xnearest = self.nearest(xrand)
            # print("after nearest")
            # print("xnearest", xnearest)
            # print("xrand", xrand)
            
            xnew, dist = self.steer(xnearest, xrand, True)

            if dist <= 0:
                self.ind += 1
                continue
            #print("xrand", xrand)
            collide = self.is_collide(xnearest, xnew, dist=dist)
            # print("after is_collide")
            if not collide:
                # print("not collide")
                self.V.append(xnew)  # add point
                self.wireup(xnew, xnearest)

                if self.getDist(xnew, self.xt) <= self.stepsize:
                    # print("getDist")
                    self.wireup(self.xt, xnew)
                    # print("after wireup")
                    _ = self.path(timeout_threshold)
                    # print('Total distance = ' + str(D))
                    return np.array(self.Path[::-1])
                    
                self.i += 1
            self.ind += 1
        # print("before return None")
        self.done = True

        print('No path find !')
        return None

    def path(self, timeout_threshold, dist=0):
        start_time = time.time()
        # print("x0 is: ", self.x0)
        # print("xt is: ", self.xt)
        x = self.xt
        self.Path.clear()

        #print('path is ',self.Path)
        while x != self.x0:
            x2 = self.Parent[x]
            self.Path.append(x)
            #print(x)
            dist += self.getDist(x, x2)
            x = x2

        self.Path.append(self.x0)

        #print('path is ', np.array(self.Path[::-1]))
        return np.array(self.Path[::-1]), dist

    def is_collide(self, x, child, dist=None):
        if dist==None:
            dist = self.getDist(x, child)
        
        result, _ = self.kdtree.query(x)
        
        if self.is_inside_obs(child) or self.is_inside_obs(x):
            return True
        
        vect = np.array(child) - np.array(x)
        num = math.ceil(dist / 0.2)
        for i in range(num+1):
            new_pt = x + (1.0 * i / (1.0 *num))* vect
            #print("new_pt ", new_pt )
            if self.is_inside_obs(new_pt):
                return True
    
        return False



    def steer(self, x, y, DIST=False):
        # steer from s to y
        if np.equal(x, y).all():
            return x, 0.0
        dist = self.getDist(y, x)
        #print("dis is ", dist)
        #print("x is ", x)
        step = min(dist, self.stepsize)
        increment = ((y[0] - x[0]) / dist * step, (y[1] - x[1]) / dist * step, (y[2] - x[2]) / dist * step)
        xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])
        #print("increment is ", increment)
        #print("xnew  is ",xnew )
        # direc = (y - s) / np.linalg.norm(y - s)
        if DIST:
            return xnew, dist
        return xnew, dist



    def sampleFree(self, bias = 0.1):
        '''biased sampling'''
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        i = np.random.random()

        #print(i)
        if self.is_inside_obs(x):
            return None
        else:
            if i < bias:
                return np.array(self.xt) + 1
            else:
                return x
            return x

    def is_inside_obs(self, x):

        result, _ = self.kdtree.query(x)
        #print(result)
        if result < self.safe_dis:
            return True
        return False


    def nearest(self, x, isset=False):
        V = np.array(self.V)
        if self.i == 0:
            return self.V[0]
        xr = repmat(x, len(V), 1)
        dists = np.linalg.norm(xr - V, axis=1)
        return tuple(self.V[np.argmin(dists)])



    def getDist(self, pos1, pos2):
        return np.sqrt(sum([(pos1[0] - pos2[0]) ** 2, (pos1[1] - pos2[1]) ** 2, (pos1[2] - pos2[2]) ** 2]))

