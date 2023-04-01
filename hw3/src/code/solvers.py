'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N, ))
    
    #compute output
    x = inv(A.T@A)@A.T@b
    
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)
    
    #perform LU factorization
    lu = splu(A.T@A, permc_spec='NATURAL')
    
    #solve for x
    x = lu.solve(A.T@b)
    
    return x, lu.U

def solve_lu_custom_solver(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)
    
    #perform LU factorization
    lu = splu(A.T@A, permc_spec='NATURAL')
        
    #find number of rows
    N = lu.L.A.shape[0]
    
    #store intermediate results
    y = np.zeros(N)
    
    #store which variables are found
    mask = np.zeros(N, 'int')
    
    #L and d in of Ly=d
    L = lu.L.A
    d = A.T@b
    
    #iterate to find all elements in y
    for i in range(N):
        #mask of non zero elements in L matrix
        mask_L = L[i] != 0
        
        #idx of unknown element among mask_L
        idx = int(np.where(np.logical_and(mask == 0, mask_L))[0])
        
        #mask of known elements
        mask_known = mask != 0
        
        #find value of unknown element at idx
        y[idx] = (d[idx] - np.dot(L[i][mask_known], y[mask_known]))/L[i][idx]
        
        #update mask
        mask[idx] = 1

    #store final results
    x = np.zeros(N)
    
    #store which variables are found
    mask = np.zeros(N, 'int')
    
    #U in Ux=y
    U = lu.U.A
    
    #iterate to find all elements in x
    for i in range(N-1, -1, -1):
        #mask of non zero elements in U matrix
        mask_U = U[i] != 0
        
        #idx of unknown element among mask_U
        idx = int(np.where(np.logical_and(mask == 0, mask_U))[0])
        
        #mask of known elements
        mask_known = mask != 0
        
        #find value of unknown element at idx
        x[idx] = (y[idx] - np.dot(U[i][mask_known], x[mask_known]))/U[i][idx]
        
        #update mask
        mask[idx] = 1
            
    return x, lu.U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)
    
    #perform LU factorization
    lu = splu(A.T@A, permc_spec='COLAMD')
    
    #solve for x
    x = lu.solve(A.T@b)
    
    return x, lu.U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)
    
    #perform factorization
    z, R, _, _ = rz(A, b)
    
    #solve for x
    x = spsolve_triangular(R, z, lower=False)
    
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)
    
    #perform factorization
    z, R, E, _ = rz(A, b, permc_spec='COLAMD')
        
    #solve for x
    x = spsolve_triangular(R, z, lower=False)
    
    # #change x
    x[E] = x.copy()
        
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'lu_custom': solve_lu_custom_solver,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)
