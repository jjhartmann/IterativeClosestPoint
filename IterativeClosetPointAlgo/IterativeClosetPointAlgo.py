import sys

sys.path.insert(1, '../Levenberg-Marquardt-Algorithm')
import LMA

from array import array
from math import sin, pi
from random import random
import numpy as np
import math


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
    R_params = [7, 20, 8]
    t_params = [-5, -9, 10]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, noise = True, mu = 0, sigma = 10)


    # Check if transform works
    invert_points, R, t =  transform(transform_parms, transfomed_points, invert = True)
    if np.array_equal(model_points, invert_points):
        print("Test Passed")

    results = ICP(model_points, transfomed_points, verbose=True)

    print("END")




if __name__ == "__main__":
    main()

