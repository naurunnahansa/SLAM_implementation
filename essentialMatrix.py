import cv2
import numpy as np 
import math  as m

class EssentialMatrix(object):
    def __init__(self, K, P):
        self.K = K
        self.P = P

    @staticmethod
    def _form_transf(R, t):
        # gives transformation matrix from the rotation and translation matricies
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_pose(self, q1, q2):
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        E = self._form_transf(R,t)
        return E
        pass

    def getYawPitchRoll(self, E):
        roll = m.atan(E[2][1]/E[2][2])
        pitch =  m.atan((-E[2][0])/m.sqrt(m.pow(E[2][1],2) + m.pow(E[2][2],2)))
        yaw =  m.atan(E[1][0]/E[0][0])
        return yaw, pitch, roll



    #Essential_Matrix -> Sb(translation) + R(relative rotation) 
    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                    np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if (max == 2):return R1, np.ndarray.flatten(-t)
        elif (max == 3): return R2, np.ndarray.flatten(-t)
        elif (max == 0):return R1, np.ndarray.flatten(t)
        elif (max == 1):return R2, np.ndarray.flatten(t)

    