'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    #MY IMPLEMENTATION
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    
    #MY IMPLEMENTATION
    #extract r and beta from measurements
    beta = init_measure[0::2, 0]
    r = init_measure[1::2, 0]

    #compute mean positions
    #x coordinates
    landmark[0::2, 0] = init_pose[0] + r*np.cos(beta + init_pose[2])
    
    #y coordinates
    landmark[1::2, 0] = init_pose[1] + r*np.sin(beta + init_pose[2])
    
    #compute covariances
    #C
    C = np.zeros((k, 2,3))
    C[:, 0, 0] = 1 
    C[:, 1, 1] = 1
    C[:, 0, 2] = -r*np.sin(beta + init_pose[2])
    C[:, 1, 2] = r*np.cos(beta + init_pose[2])

    #D
    D = np.zeros((k, 2, 2))
    D[:,0,0] = -r*np.sin(beta + init_pose[2])
    D[:,0,1] = np.cos(beta + init_pose[2])
    D[:,1,0] = r*np.cos(beta + init_pose[2])
    D[:,1,1] = np.sin(beta + init_pose[2])
    
    #landmark covariances
    landmark_cov_ = 0*C@init_pose_cov[np.newaxis]@C.transpose((0, 2, 1)) + \
                    D@init_measure_cov[np.newaxis]@D.transpose((0, 2, 1)) #k x 2 x 2
    
    #reshape covariance vector
    for idx in range(k):
        landmark_cov[2*idx:2*idx+2, 2*idx:2*idx+2] = landmark_cov_[idx]
    
    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    #MY IMPLEMENTATION
    #make a copy of arrays
    X_new = X.copy()
    P_new = P.copy()
    
    #update state based on control data
    X_new[0] = X[0] + control[0]*np.cos(X[2])
    X_new[1] = X[1] + control[0]*np.sin(X[2])
    X_new[2] = warp2pi(X[2] + control[1])
    
    #update covariance
    #initialize Jacobian matrices
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    
    #compute Jacobian matrices
    A[0,0] = 1
    A[0,2] = -control[0]*np.sin(X[2])
    A[1,1] = 1
    A[1,2] = control[0]*np.cos(X[2])
    A[2,2] = 1
    
    B[0,0] = np.cos(X[2])
    B[0,1] = -np.sin(X[2])
    B[1,0] = np.sin(X[2])
    B[1,1] = np.cos(X[2])
    B[2,2] = 1
    
    #compute output covariance
    P_new[0:3, 0:3] = A@P[0:3,0:3]@A.T + B@control_cov@B.T
        
    return X_new, P_new


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    #MY IMPLEMENTATION
    #make a copy of arrays
    X_new = X_pre.copy()
    P_new = P_pre.copy()
    
    #extract positions of landmarks
    #lx
    lx = X_pre[3::2,0]
    
    #ly
    ly = X_pre[4::2,0]
    
    #extract l_x-x_t and l_y-y_t 
    dx = lx - X_pre[0,0]
    dy = ly - X_pre[1,0]
    
    #Compute H_p, H_l and H_t
    #initialize
    H_p = np.zeros((2*k, 3))
    H_l = np.zeros((2*k, 2*k))
    H_t = np.zeros((2*k, 3+2*k))
    
    #H_p
    H_p[0::2, 0] = dy/(dx**2 + dy**2)
    H_p[0::2, 1] = -dx/(dx**2 + dy**2)
    H_p[0::2, 2] = -1
    H_p[1::2, 0] = -dx/np.sqrt(dx**2 + dy**2)
    H_p[1::2, 1] = -dy/np.sqrt(dx**2 + dy**2)
    
    #H_l
    for idx in range(k):
        H_l[2*idx, 2*idx] = -dy[idx]/(dx[idx]**2 + dy[idx]**2)
        H_l[2*idx, 2*idx+1] = dx[idx]/(dx[idx]**2 + dy[idx]**2)
        H_l[2*idx+1, 2*idx] = dx[idx]/np.sqrt(dx[idx]**2 + dy[idx]**2)
        H_l[2*idx+1, 2*idx+1] = dy[idx]/np.sqrt(dx[idx]**2 + dy[idx]**2)
        
    #H_t
    H_t = np.hstack((H_p, H_l))
    
    #Populate Q matrix
    Q_t = np.zeros((2*k, 2*k))
    for idx in range(k):
        Q_t[2*idx:2*idx+2, 2*idx:2*idx+2] = measure_cov
    
    #Compute Kalman Gain
    K_t = P_pre@H_t.T@np.linalg.inv(H_t@P_pre@H_t.T + Q_t)
    
    #Compute expected measurement vector
    z_exp = np.zeros((2*k,1))
    z_exp[0::2,0] = warp2pi(np.arctan2(dy, dx) - X_pre[2,0])
    z_exp[1::2,0] = np.sqrt(dx**2 + dy**2)
    
    #compute error vector
    z_err = measure - z_exp
    
    #compute mean of state vector
    X_new = X_pre + K_t@z_err
    
    #compute covariance
    P_new = (np.eye(3 + 2*k) - K_t@H_t)@P_pre
    
    return X_new, P_new

def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)
    
    #MY IMPLEMENTATION
    #reshape
    l_gt = np.reshape(l_true, (-1, 2, 1))
    l_pred = np.reshape(X[3:,0], (-1, 2, 1))
    l_sigma = np.zeros((k, 2, 2))
    for idx in range(k):
        l_sigma[idx] = P[3+2*idx : 3+2*idx+2, 3+2*idx : 3+2*idx+2]
        
    #eucledian distance
    e_dist = np.sqrt(np.sum((l_pred - l_gt)**2, axis=1)).flatten()
    
    #mahalanobis distance
    m_dist = (l_pred - l_gt).transpose((0, 2, 1))@l_sigma@(l_pred - l_gt)
    m_dist = np.sqrt(m_dist).flatten()
    
    print('Euclidean distance')
    print(e_dist)
    print('Mahalanobis distance')
    print(m_dist)
    


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01;
    sig_r = 0.08;


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)
            
        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1
                        
    plt.savefig('result.png')
    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)

if __name__ == "__main__":
    main()
