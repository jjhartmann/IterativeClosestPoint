import sys

sys.path.insert(1, '../Levenberg-Marquardt-Algorithm')
import LMA

from array import array
from math import sin, pi
from random import random
import numpy as np
import math



############################################################################
# DRAW UTIL
############################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib, time



class plot3dClass( object ):

    def __init__( self, M, S,  marker=['o', '^'] ):
        self.marker = marker

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        #self.ax.set_zlim3d( -10e-9, 10e9 )

        self.scatter1 = self.ax.scatter(M[0], M[1], M[2], marker=self.marker[0])
        self.scatter2 = self.ax.scatter(S[0], S[1], S[2],   marker=self.marker[1])
        plt.draw() 

    def drawNow( self, M, S ):
        self.ax.clear()
        #self.scatter1.remove()
        #self.scatter2.remove()
        self.scatter1 = self.ax.scatter(M[0], M[1], M[2], marker=self.marker[0])
        self.scatter2 = self.ax.scatter(S[0], S[1], S[2],   marker=self.marker[1])
        
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)



def Plot3D(X, marker='o'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[0], X[1], X[2], marker=marker)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def ComparePointCloud3D(M, S, marker=['o', '^']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(M[0], M[1], M[2], marker = marker[0])
    ax.scatter(S[0], S[1], S[2], marker = marker[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



############################################################################
############################################################################


def transform(params, pointset, invert=False, noise=False, mu=0, sigma=10):
    """
    Transforma a point set into another corrdinate system
    :param params: contains rotation [0:3] and translation [3:6]
    :param pointset: a set of points to transform
    :param mu: variation 
    :param sigma:
    :return:

    See: https://math.stackexchange.com/questions/1234948/inverse-of-a-rigid-transformation
    """

    thetas = np.array(params[0:3]) * math.pi/180
    t = params[3:6]

    Rx = np.eye(3, 3)
    Rx[1, 1] = np.cos(thetas[0])
    Rx[1, 2] = -np.sin(thetas[0])
    Rx[2, 1] = np.sin(thetas[0])
    Rx[2, 2] = np.cos(thetas[0])

    Ry = np.eye(3, 3)
    Ry[0, 0] = np.cos(thetas[1])
    Ry[2, 0] = -np.sin(thetas[1])
    Ry[0, 2] = np.sin(thetas[1])
    Ry[2, 2] = np.cos(thetas[1])

    Rz = np.eye(3, 3)
    Rz[0, 0] = np.cos(thetas[2])
    Rz[0, 1] = -np.sin(thetas[2])
    Rz[1, 0] = np.sin(thetas[2])
    Rz[1, 1] = np.cos(thetas[2])

    R = Rz @ Ry @ Rx

    if invert:
        R = R.transpose()
        t = np.multiply(-1, t).transpose() @ R 

    transformed_points = np.empty(shape=pointset.shape, dtype=np.float)
    if noise:
        point_s = pointset + np.random.normal(mu, sigma, pointset.shape)
        transformed_points = point_s.transpose() @ R
    else:
        transformed_points = pointset.transpose() @ R

    transformed_points[:, 0] += t[0]
    transformed_points[:, 1] += t[1]
    transformed_points[:, 2] += t[2]

    return transformed_points.transpose(), R, t


def icp_error_function(params, args):
    """
    Simple error function to determine error between pointsets
    """

    # Pointsets
    M, S = args

    # Transform point set
    S_T, R, t = transform(params, S, invert=True)

    tmp = M - S_T
    l2Error = np.linalg.norm(tmp.transpose(), axis=1)

    # flatten arays
    dataShape = M.shape
    nData = dataShape[0] * dataShape[1]

    M_hat = M.reshape(1, nData)[0]
    S_That = S_T.reshape(1, nData)[0]

    absError = np.abs(M_hat - S_That)

    return l2Error



############################################################################
# ICP - Iterative Closest Point
############################################################################
def ICP(M, S, verbose = False):
    """ Perform Simple Point Set Registration
    :param M: Base Point Set
    :param S: Point Set to match
    :return: params the best transforms S to match M
    """

    if verbose:
       ICP.plot = plot3dClass(M, S)

    params = np.zeros(6) # np.random.rand(6) * 50
    registered = False
    count = 0
    while not registered:
        
        X = np.zeros(S.shape)
        S_T, _, _ = transform(params, S, invert=True)
        S_T = S_T.transpose()

        # Iterate through arrays
        for i, s_i in enumerate(S_T):
            
            minDist = 100000
            p = np.zeros(3)

            # Find closest point
            for m_i in M.transpose():
                d = np.linalg.norm(m_i - s_i)
                if d < minDist:
                    p = m_i
                    minDist = d

            # Add pair to array
            X[:, i] = p


        # Update params using LMA
        rmserror, params, reason = LMA.LM(
            params, 
            (X, S),
            icp_error_function,
            lambda_multiplier=10,  
            kmax=100, 
            eps=1e-3)

        if verbose:
            ICP.plot.drawNow(M, S_T.T)
            print("{} RMS: {} Params: {}".format(count, rmserror, params))

        if rmserror < 1e-2:
            registered = True

        count = count + 1

    return params





############################################################################
# Main
############################################################################

def TestICP():
    # Generate Model Points
    model_points = ((2 * np.random.rand(3, 50)) - 1) * 500

    # Ground truth transformation parameters
    #           x    y   z
    R_params = [30, 20, -20]
    t_params = [30,-100, 90]#[-50, -90, 100]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, noise = False, mu = 0, sigma = 10)

    # Visualizat Pointsets
    #ComparePointCloud3D(model_points, transfomed_points)

    # Check if transform works
    invert_points, R, t =  transform(transform_parms, transfomed_points, invert = True)
    if np.array_equal(model_points, invert_points):
        print("Test Passed")


    # Optimize Transform Params
    results = ICP(model_points, transfomed_points, verbose=True)

    # Check transfomed points
    result_points, _, _ = transform(results, transfomed_points, invert=True)
    #ComparePointCloud3D(model_points, result_points)


def main():

    # Test regular iterative closest point (ICP)
    #TestICP()
    

    """
    Test Deterministic Annealing
    Gold, S., Rangarajan, A., Lu, C. P., Pappu, S., & Mjolsness, E. (1998). 
    New algorithms for 2D and 3D point matching: Pose estimation and correspondence. 
    Pattern Recognition, 31(8), 1019–1031. https://doi.org/10.1016/S0031-3203(98)80010-1
    """

    M = np.zeros(10)
    Q = np.random.rand(10)


    B = 1
    while B < 1000:
        
        q_sum = 0
        for q in Q:
            q_sum = q_sum + math.exp(B * q)

        for i, q in enumerate(Q):
            M[i] = math.exp(B * q) / q_sum

        B = B * 2
        print("{} M: {} ".format(B, M))

    for i, m in enumerate(M):
        if m >= 0.99:
            print("Index: {} Max: {}".format(i, Q[i]))


    print("END")




if __name__ == "__main__":
    main()

