import numpy as np

def list_to_mat(data, dims):
    m, n = dims
    n_obs = len(data[:, 0])
    out1 = np.zeros((m, n))
    out2 = np.zeros((m, n))

    for ind in range(n_obs):
        i, j = data[ind, :2]
        i = int(i)
        j = int(j)
        out1[i, j] = data[ind, -1]
        out2[i, j] = 1

    return out1, out2


def predict(test_set, U, V):
    n_test = test_set.shape[0]
    i_obs = test_set[:, 0].astype('int')
    j_obs = test_set[:, 1].astype('int')

    UV_obs = np.sum(U[i_obs, :] * V[:, j_obs].T, axis=1)
    diff = (test_set[:, -1] - UV_obs)
    count = np.sum(np.abs(diff) >= 1)
    return (np.sum(diff ** 2) ** 0.5, count / len(i_obs))


def log_joint(U, V, X_list, gamma_U_params, gamma_V_params):
    m, d = np.shape(U)
    _, n = np.shape(V)
    A_u, B_u = gamma_U_params['a'], gamma_U_params['b']
    A_v, B_v = gamma_V_params['a'], gamma_U_params['b']
    n_obs = len(X_list[:, 0])
    i_obs = X_list[:, 0].astype('int')
    j_obs = X_list[:, 1].astype('int')
    rel_UV = np.sum(U[i_obs, :] * V[:, j_obs].T, axis=1)
    pt_poisson = np.sum(X_list[:, 2] * np.log(rel_UV) - rel_UV)

    return pt_poisson
