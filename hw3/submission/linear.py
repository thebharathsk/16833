'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import os
import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def create_linear_system(odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2
    offset_y = (n_odom + 1) * 2
    offset_x = n_poses * 2

    A = np.zeros((M, N))
    B = np.zeros((M, ))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))
    
    #MY IMPLEMENTATION
    #initialize Jacobian
    H = np.zeros((2,4))
    H[0,0] = -1
    H[1,1] = -1
    H[0,2] = 1
    H[1,3] = 1
    
    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    A[0, 0] = 1
    A[1, 1] = 1
    
    # TODO: Then fill in odometry measurements
    for i, odom in enumerate(odoms):
        #reshape odomometry reading
        odom = odom.flatten()

        #find a and b
        a = sqrt_inv_odom@H
        b = sqrt_inv_odom@odom
        
        #add a and b to A and B
        A[(i+1)*2:(i+2)*2, i*2:(i+2)*2] = a
        B[(i+1)*2:(i+2)*2] = b
        
    # TODO: Then fill in landmark measurements
    for i, obs in enumerate(observations):
        #read indices
        m, n = int(obs[0]), int(obs[1])
        
        #reshape observation
        obs = obs[2:].flatten()

        #find a and b
        a = sqrt_inv_obs@H
        b = sqrt_inv_obs@obs
        
        #add a and b to A and B
        A[offset_y + i*2 : offset_y + (i+1)*2, m*2:(m+1)*2] = a[:,0:2]
        A[offset_y + i*2 : offset_y + (i+1)*2, offset_x + n*2:offset_x + (n+1)*2] = a[:,2:]
        B[offset_y + i*2 : offset_y + (i+1)*2] = b
    
    return csr_matrix(A), B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='path to npz file')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd', 'lu_custom'],
        default=['default'],
        help='method')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help=
        'Number of repeats in evaluation efficiency. Increase to ensure stablity.'
    )
    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    # plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt trajectory')
    # plt.scatter(gt_landmarks[:, 0],
    #             gt_landmarks[:, 1],
    #             c='b',
    #             marker='+',
    #             label='gt landmarks')
    # plt.legend()
    # plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Build a linear system
    A, b = create_linear_system(odoms, observations, sigma_odom,
                                sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f'Applying {method}')

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f'{method} takes {total_time / total_iters}s on average')

        if R is not None:
            plt.spy(R)
            #plt.show()
            
            #save plot
            #create file name
            file = os.path.join('./../../report/results/', \
                                args.method[0]+ '_'+\
                                os.path.basename(args.data).split('.npz')[0] + '_sparsity.png')
            #plt.savefig(file)
            plt.close()

        traj, landmarks = devectorize_state(x, n_poses)
        
        # Visualize the final result
        #create file name
        file = os.path.join('./../../report/results/', \
                            args.method[0]+ '_'+\
                            os.path.basename(args.data).split('.npz')[0] + '_map.png')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, save_path=None)
