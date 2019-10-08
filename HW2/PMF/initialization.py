import numpy as np
import scipy.stats as spst

def rand_col_init(X, d):
    m, n = X.shape
    d = d
    shuffled_n = list(range(n))
    np.random.shuffle(shuffled_n)
    U = np.zeros((m, d))
    set_size = int(m / d)

    for i in range(d):
        try:
            U[:, i] = np.sum(X[:, shuffled_n[int(i * set_size):int((i + 1) * set_size)]], axis=1)
        except IndexError:
            U[:, i] = np.sum(X[:, shuffled_n[int(i * set_size):]], axis=1)

    for i in range(d):
        bool_slice = U[:, i] == 0
        n_neq0 = sum(bool_slice)
        U[bool_slice, i] = spst.uniform.rvs(0, 5, size=n_neq0)

    return U