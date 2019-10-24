import sys

sys.path.insert(1, '../Levenberg-Marquardt-Algorithm')
import LMA

from array import array
from math import sin, pi
from random import random
import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D 
from scipy.misc import imshow
import matplotlib.pyplot as plt
import pptk 

############################################################################
# DRAW UTIL
############################################################################

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



def StartPlot(M, S, marker=['o', '^']):
    StartPlot.v = pptk.viewer((M, S))
    

    attr1 = np.zeros(np.max(M.shape))
    attr2 = np.ones(np.max(S.shape))
    attr = np.concatenate((attr1, attr2), axis=None)
    
    StartPlot.v.attributes(attr)
    StartPlot.v.set(point_size=5)

    StartPlot.counter = 0


def Update3DCompare(M, S, marker=['o', '^']):

    StartPlot.counter = StartPlot.counter + 1
    if StartPlot.counter % 5 > 0:
        return

    points = np.concatenate((M, S.T), axis=1)

    attr1 = np.zeros(np.max(M.shape))
    attr2 = np.ones(np.max(S.shape))
    attr = np.concatenate((attr1, attr2), axis=None)

    positions = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    StartPlot.v.__load(positions)





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


def ICP(M, S, verbose = False):
    """ Perform Simple Point Set Registration
    :param M: Base Point Set
    :param S: Point Set to match
    :return: params the best transforms S to match M
    """

    if verbose:
        StartPlot(M, S)

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
            Update3DCompare(M, S_T)
            print("{} RMS: {} Params: {}".format(count, rmserror, params))

        if rmserror < 1e-2:
            registered = True

        count = count + 1

    return params




def main():
    
    # Generate Model Points
    model_points = ((2 * np.random.rand(3, 100)) - 1) * 500

    # Ground truth transformation parameters
    #           x    y   z
    R_params = [30, 20, 8]
    t_params = [3,-10,2]#[-50, -90, 100]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, noise = False, mu = 0, sigma = 10)

    # Visualizat Pointsets
    #ComparePointCloud3D(model_points, transfomed_points)


    #v = pptk.viewer((model_points, transfomed_points))
    

    #attr1 = np.zeros(np.max(model_points.shape))
    #attr2 = np.ones(np.max(transfomed_points.shape))
    #attr = np.concatenate((attr1, attr2), axis=None)
    
    #v.attributes(attr)
    #v.set(point_size=5)
    #v.wait()

    # Check if transform works
    invert_points, R, t =  transform(transform_parms, transfomed_points, invert = True)
    if np.array_equal(model_points, invert_points):
        print("Test Passed")


    # Optimize Transform Params
    results = ICP(model_points, transfomed_points, verbose=True)

    # Check transfomed points
    result_points, _, _ = transform(results, transfomed_points, invert=True)
    ComparePointCloud3D(model_points, result_points)


    print("END")




if __name__ == "__main__":
    main()

