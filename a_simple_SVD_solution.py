import numpy as np
from numpy.linalg import eig


def SVD(A):
    '''
    A = U*sigma*transpose(V)
    :param A: The Matrix A
    :return:
    '''
    m = A.shape[0]
    n = A.shape[1]
    A_T = A.transpose() # n*m
    V_T = np.matmul(A_T,A) # n*n
    U = np.matmul(A,A_T) # m*m
    v_eigens,_ = matrix_decomposition(V_T) # n eigen values
    u_eigens,_ = matrix_decomposition(U) # m eigen values
    sigma_diags = u_eigens if n > m else v_eigens
    sigma = np.diag(sigma_diags)
    padding = np.zeros([m-n,n]) if m > n else np.zeros([n-m,m])
    sigma = np.concatenate([sigma,padding])
    return U,sigma,V_T

def matrix_decomposition(mat):
    '''

    :param mat: the matrix
    :return: eigen values and eigen vectors
    '''
    w,v = eig(mat)
    return w,v

if __name__ == '__main__':
    A = np.ones([3,2]).astype('int32')
    A[0][0],A[2][1] = 0,0
    U,sigma,V_T = SVD(A)
    pass
