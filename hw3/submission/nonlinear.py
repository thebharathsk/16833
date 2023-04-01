'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
import os
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=np.bool_)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    '''
    # TODO: return odometry estimation
    odom = (x[(i+1)*2:(i+2)*2] - x[i*2:(i+1)*2]).flatten()

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    obs = np.zeros((2, ))

    #read pose, landmark and position data
    pose = x[i*2:(i+1)*2]
    landmark = x[n_poses*2 + j*2: n_poses*2 + (j+1)*2]
    rel_pos = landmark - pose
    rel_pos = rel_pos.flatten()
    
    #compute bearing angle and distance 
    obs[0] = warp2pi(np.arctan2(rel_pos[1], rel_pos[0]))
    obs[1] = np.linalg.norm(rel_pos)
    
    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))

    #read pose, landmark and position data
    pose = x[i*2:(i+1)*2]
    landmark = x[n_poses*2 + j*2 : n_poses*2 + (j+1)*2]
    rel_pos = landmark - pose
    rel_pos = rel_pos.flatten()
    
    #compute useful variables
    dx = rel_pos[0]
    dy = rel_pos[1]
    dx_2 = dx**2
    dy_2 = dy**2
    
    #populate jacobian
    jacobian[0] = [dy, -dx, -dy, dx]/(dx_2+dy_2)
    jacobian[1] = [-dx, -dy, dx, dy]/np.sqrt(dx_2+dy_2)
    
    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
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
    
    B[0:2] = -x[0:2]
    
    # TODO: Then fill in odometry measurements
    for i, odom in enumerate(odoms):
        #reshape odomometry reading
        odom = odom.flatten()

        #estimate expected measurement at x
        f_x_0 = odometry_estimation(x, i)
        
        #compute measurement error
        odom_err = odom - f_x_0
        
        #find a and b
        a = sqrt_inv_odom@H
        b = sqrt_inv_odom@odom_err
        
        #add a and b to A and B
        A[(i+1)*2:(i+2)*2, i*2:(i+2)*2] = a
        B[(i+1)*2:(i+2)*2] = b
        
    # TODO: Then fill in landmark measurements
    for i, obs in enumerate(observations):
        #read indices
        m, n = int(obs[0]), int(obs[1])
        
        #reshape observation
        obs = obs[2:].flatten()

        #estimate expected measurement at x
        g_x_0 = bearing_range_estimation(x, m, n, n_poses)
        
        #compute measurement error
        obs_err = obs - g_x_0
        
        #use warp angle
        obs_err[0] = warp2pi(obs_err[0])
        
        #estimate jacobian
        J = compute_meas_obs_jacobian(x, m, n, n_poses)
        
        #find a and b
        a = sqrt_inv_obs@J
        b = sqrt_inv_obs@obs_err
        
        #add a and b to A and B
        A[offset_y + i*2 : offset_y + (i+1)*2, m*2:(m+1)*2] = a[:,0:2]
        A[offset_y + i*2 : offset_y + (i+1)*2, offset_x + n*2:offset_x + (n+1)*2] = a[:,2:]
        B[offset_y + i*2 : offset_y + (i+1)*2] = b

    return csr_matrix(A), B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='../data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx
        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        
        #create file name
        file = os.path.join('./../../report/results/', \
                            args.method[0]+ '_'+\
                            os.path.basename(args.data).split('.npz')[0] + '_map.png')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, save_path=None)
