from PMF.initialization import *
from PMF.utils import *
import time as tm
import scipy.sparse as spsp
import numpy as np


def update_matrix_fast(U, V, X_list, gamma_U_params):
    """
    we are doing the following problem:
    lambda_u -> u_{i,:} -> x_ij <- (v^*)_{j,:} <- lambda_v

    U: a matrix of shape (m,d) (or of shape (n,d) if V.T is the focus)
    V: a matrix of shape (d,n) (or of shape (m,d) if U.T is here)
    X: a matrix of shape (m,n), possibly has missing value represented
       by 0
    gamma_U_params: a dict, looks like {'a','b'}
    """
    A, B = gamma_U_params['a'], gamma_U_params['b']
    m, d = U.shape
    _, n = V.shape
    n_obs = X_list.shape[0]
    UV_sp = np.zeros(n_obs)
    ind_i = X_list[:, 0].astype('int')
    ind_j = X_list[:, 1].astype('int')
    UV_sp = np.sum(U[ind_i, :] * (V[:, ind_j].T), axis=1)
    X_UV = spsp.csc_matrix((X_list[:, -1] / UV_sp, (ind_i, ind_j)), shape=(m, n))
    M = spsp.csc_matrix((np.ones(n_obs), (ind_i, ind_j)), shape=(m, n))

    part1 = (A + np.multiply(U, (X_UV @ (V.T))))
    part2 = (A / B + M @ (V.T))

    U_new = part1 / part2
    return (U_new)


def Poisson_pmf(data ,dims ,low_rank ,U_params={'a' :1 ,'b' :1}, V_params={'a' :1 ,'b' :1}, T=100 ,verbose = False
                ,full=False):
    m ,n = dims
    d = low_rank
    n_obs = len(data[: ,0])

    data_T = data[: ,[1 ,0 ,2]]

    mat_data, mat_mask = list_to_mat(data ,dims)
    U_0 =rand_col_init(mat_data ,d)
    V_0 =rand_col_init(mat_data.T ,d).T

    if full:
        log_post_record = np.zeros(T)
    U = U_0
    V = V_0
    for t in range(T):
        start = tm.time()
        U = update_matrix_fast(U ,V ,data ,U_params)
        mid = tm.time()
        V = update_matrix_fast(V.T ,U.T ,data_T ,V_params).T
        mid2 = tm.time()
        if full:
            log_post_record[t] = log_joint(U ,V ,data ,U_params ,V_params)
        end = tm.time()
        if verbose:
            print('=======  ' +str(t ) +' =======')
            print("speed break down: \n")
            print("U_update: \t" +str(mid - start) + 's')
            print("V_update: \t" + str(mid2 - mid) + 's')
            print("log posterior: \t" + str(end - mid2) + 's')
            # print("matrix norm of difference")
            # print(np.linalg.norm((U@V)*mat_mask-mat_data*mat_mask))
            if full:
                print("log posterior:")
                print(log_post_record[t])

    if full:
        return (U, V, log_post_record)
    else:
        return (U, V)