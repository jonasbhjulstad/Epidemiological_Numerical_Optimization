import numpy as np
from itertools import combinations, chain, tee
from FROLS_Scripts import
import pandas as pd


def generate_data(magnitude=1, loc=0, sigma=1, span=[-3,3], Nu = 200, param=[2, 3, 2, 2]):
    u = np.random.uniform(span[0], span[1], Nu)

    true_model = bilinear_term(param)

    max_y_lag = param[1]
    max_u_lag = param[0]
    print(true_model)
    y_true = np.zeros(max_y_lag + Nu)
    y = np.zeros(max_y_lag + Nu)
    for i in range(u.shape[0]-max_u_lag):
        ik = i + max_y_lag
        y_prev = y_true[i:ik].reshape((-1,1))
        u_prev = u[i:ik].reshape((-1,1))
        Xk = np.concatenate([u_prev, y_prev], axis=1)
        y_true[ik] = true_model(Xk)
        y[ik] = y_true[ik] + magnitude * np.random.laplace(loc, sigma)

    X = np.concatenate([u, y[:-max_y_lag]], axis=1)

    return true_model, X, y, y_true, u

if __name__ == '__main__':

    # np.seterr(all='raise')




    true_model, X, y, y_true, u = generate_data()

    X_LS = X[:len(y), :]
    LS_params = np.linalg.inv(X_LS.T @ X_LS) @ (X_LS.T @ y)

    rho = 0.0000001

    g, q, ERR_list, ells, A, p = FROLS(X, y, rho)


