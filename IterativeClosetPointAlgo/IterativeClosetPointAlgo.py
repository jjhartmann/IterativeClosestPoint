import sys

sys.path.insert(1, '../Levenberg-Marquardt-Algorithm')
import LMA

from array import array
from math import sin, pi
from random import random
import numpy as np
import math
import plyfile


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



############################################################################
# ICP - Iterative Closest Point
############################################################################

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
            kmax=10, 
            eps=1e-3)

        if verbose:
            ICP.plot.drawNow(M, S_T.T)
            print("{} RMS: {} Params: {}".format(count, rmserror, params))

        if rmserror < 1e-2:
            registered = True

        count = count + 1

    return params




############################################################################
# RPM - Robust Point Matching 
############################################################################

def rpm2D_cost_function(params, args):
    """
    RMP Cost Function
    
    cost = sum M sum S(
    """

    alpha = 1

    # Pointsets
    M, S, MMatrix = args

    # Transform point set
    S_hat, R, t = transform(params, S, invert=True)

    # Build array
    cost = np.zeros(np.max(M.shape) * np.max(S.shape))

    k = 0
    for (i, j), mm in np.ndenumerate(MMatrix):
        cost[k] = mm * ((np.linalg.norm(M.T[i]- S_hat.T[j]) - alpha)) # - regularizing term (todo)

        k = k + 1

    return cost



def RPM3D(M, S, B0=0.01, Bf=1.01, Bmax = 500, gamma0=1e-03, gammaf=1.2, maxIter0=100, alphaTol=1, verbose=False):

    if verbose:
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        RPM3D.plot = plot3dClass(M, S)

    # Transformation (Affine) Params
    params = np.zeros(6)
    
    # Match matrix (N x M) matrix based on input (plus slack)
    MMatrix = np.ones((np.max(M.shape), np.max(S.shape)))
    Q = np.zeros((np.max(M.shape), np.max(S.shape)))

    # Deterministic Annealing Loop
    B = B0
    while B < Bmax:

        # Update Q (deriviative of cost)
        S_hat, _, _ = transform(params, S, invert=True)
        for (i, j), val in np.ndenumerate(Q):
            # dcost/dm_ij = - (|| M_i - T(S_j)||^2 - alpha)
            # alpha is the tolerence the system has towards outliers (bigger == larger tolerence)
            if j < np.max(S_hat.shape) and i < np.max(M.shape):
                Q[i, j] = - (np.linalg.norm(M.T[i] - S_hat.T[j]) - alphaTol) 
            else:
                Q[i,j] = -1

                
        # Update match matrix using sinkhorns
        for (i,j), q in np.ndenumerate(Q):
            MMatrix[i,j] = np.exp(B * q)

        M0 = np.copy(MMatrix)
        M1 = np.zeros(MMatrix.shape)
        first = True;
        it = 0
        while (not np.allclose(M0, M1, atol=0.05) or first) and it < maxIter0:
            # Update M1 with M0 (Row Normalization)
            for i, row in enumerate(M0):
                # Sum row
                row_sum = 0
                for el in row:
                    row_sum = row_sum + el
                # Update M1
                for j, el in enumerate(row):
                    M1[i, j] = el / row_sum
            # Update M0 with M1 (Col Normalization)
            for j, col in enumerate(M1.T):
                # Sum col
                col_sum = 0
                for el in col:
                    col_sum = col_sum + el
                # Update M0
                for i, el in enumerate(col):
                    M0[i, j] = el / col_sum
            first = False
            it = it + 1

       
        # Assign converged matrx to Match matrix
        MMatrix = M0

        # Use LM to optimize params
        rmserror, params1, reason = LMA.LM(
            params, 
            (M, S, MMatrix),
            rpm2D_cost_function,
            lambda_multiplier=10,  
            kmax=10, 
            eps=1e-3)
        params = params1

        if verbose:
            RPM3D.plot.drawNow(M, S_hat)
            print("B: {} \nParams: {} \n{}\n".format(B, params, MMatrix))

        # Increase heat (annealing)
        B = B * Bf


############################################################################
# Main
############################################################################

def TestICP():
    # Generate Model Points
    model_points = ((2 * np.random.rand(3, 10)) - 1) * 500

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



def DeterministicAnnealingUsingSinkhornTest(size = (5,5)):
    """
    Deterministic Annealing Test 
    See: Gold, S., Rangarajan, A., Lu, C. P., Pappu, S., & Mjolsness, E. (1998). 
    New algorithms for 2D and 3D point matching: Pose estimation and correspondence. 
    Pattern Recognition, 31(8), 1019–1031. https://doi.org/10.1016/S0031-3203(98)80010-1

    Page 10.

    Result is a a doubly stochastic matrix
    See: https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
    """
    rtol = 1e-06
    atol = 1e-03
    
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    Q = -np.random.rand(size[0], size[1]) * 100
    M = np.zeros(size)

    B = 1
    Bf = 1.1

    while B < 500:
        #Update M using softmax
        for (i,j), q in np.ndenumerate(Q):
            M[i,j] = np.exp(B * q)

        # Begin Sinkhorns Method
        M0 = np.copy(M)
        M1 = np.zeros(size)
        while not np.allclose(M0, M1, atol=atol):
            # Update M1 with M0 (Row Normalization)
            for i, row in enumerate(M0):
                # Sum row
                row_sum = 0
                for el in row:
                    row_sum = row_sum + el
                # Update M1
                for j, el in enumerate(row):
                    M1[i, j] = el / row_sum
            # Update M0 with M1 (Col Normalization)
            for j, col in enumerate(M1.T):
                # Sum col
                col_sum = 0
                for el in col:
                    col_sum = col_sum + el
                # Update M0
                for i, el in enumerate(col):
                    M0[i, j] = el / col_sum


        # Typically we would change Q here based on some output from the calculation in ICP
        # Slowly increasing B to get stronger and stronger correspondance between two point sets

        print("B: {}  \n{}\n".format(B, M0))

        #Increase B
        B = B * Bf






def main():
    #TestICP()
    #DeterministicAnnealingUsingSinkhornTest()

    # Get data
    data = plyfile.PlyData.read('sample.ply')['vertex']
    xyz = np.c_[data['x'], data['y'], data['z']]

    # get subsample
    idx = np.random.randint(np.max(xyz.shape), size=30)
    M = xyz[idx,:]
    M = M.T

    # Generate Model Points
    #M = ((2 * np.random.rand(3, 10)) - 1) * 50

    # Ground truth transformation parameters
    #           x    y   z
    R_params = [-40, 40, -20]
    t_params = [30,-100, 90]#[-50, -90, 100]
    transform_parms =  R_params + t_params
    S, R, t = transform(transform_parms, M, noise = False, mu = 0, sigma = 10)

    tmp = S.T
    np.random.shuffle(tmp)
    S = tmp.T

    # Optimize Transform Params
    results = RPM3D(M, S, verbose=True)



    print("END")




if __name__ == "__main__":
    main()

