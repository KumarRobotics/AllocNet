import numpy as np
import scipy
import math

class Trajectory:

    def __init__(self, coeffs, T):
    
        # results
        self.coeffs =coeffs
        self.T = T


        # resize the coefficients
        self.seg = len(T)
        self.dim = self.coeffs[0].shape[0]
        self.dur = 0
        self.degree   = self.coeffs[0].shape[1] - 1

    
        # print("seg is,", self.seg)
        # print("dim is,", self.dim)

        for t in T:
            self.dur += t


        print("total duration is,", self.dur)


    def get_index(self, t):

        
        for idx in range(0,  self.seg):

            dur = self.T[idx]

            if t > dur:
                t -= dur
            else:
                return t, idx

        if t > 0:
            return self.dur, self.seg-1



    # one continuity constraints    p_i(t_i) = p_{i+1}(0)
    def get_pos(self, t):

        t_on_seg, i = self.get_index(t)
     
        coeffMat = self.coeffs[i]
        pos = np.zeros(self.dim)
       
        tn = 1.0
        #print(coeffMat.shape )
        for i in range(self.degree, -1, -1):
            pos += tn * coeffMat[:, i]
            tn *= t_on_seg
        
        return pos

    def get_vel(self, t):

        t_on_seg, i = self.get_index(t)
     
        coeffMat = self.coeffs[i]
        
        vel = np.zeros(self.dim)
        tn = 1.0
        n = 1
        for i in range(self.degree-1, -1, -1):
            vel += n * tn * coeffMat[:, i]
            tn  *= t_on_seg
            n   += 1

        return vel
    

    def get_acc(self, t):


        t_on_seg, i = self.get_index(t)
     
        coeffMat = self.coeffs[i]
        #print(coeffMat)
        acc = np.zeros(self.dim)
        tn = 1.0
        m = 1
        n = 2
        for i in range(self.degree-2, -1, -1):
            acc += m * n * tn * coeffMat[:, i]
            tn  *= t_on_seg
            m   += 1
            n   += 1

        return acc

    def get_sample_pts(self, res = 0.2):

        pts = []
        for t in np.arange(0.0, self.dur, res):
          pts.append(self.get_pos(t))

        return np.asarray(pts)


    def get_sample_vels(self, res = 0.2):

        pts = []
        for t in np.arange(0.0, self.dur, res):
          pts.append(self.get_vel(t))

        return np.asarray(pts)
    
    def get_sample_accs(self, res = 0.2):

        pts = []
        for t in np.arange(0.0, self.dur, res):
            pts.append(self.get_acc(t))

        return np.asarray(pts)
    