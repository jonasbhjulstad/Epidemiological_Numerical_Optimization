import numpy as np
from itertools import islice, tee, combinations, chain
import logging
# logging.basicConfig(filename='../data/FROLS.log', encoding='utf-8', level=logging.DEBUG)

def bilinear_term_generator(param, Nx, N_samples):
    lag_max = param[0]
    order_max = param[1]
    order_list = combinations(np.concatenate([range(order_max+1)]*Nx), Nx)
    lag_list = combinations(np.concatenate([range(lag_max)]*Nx), Nx)

    for i, (lags, orders) in enumerate(zip(lag_list, order_list)):
        if np.any(lags):
            yield ([list(lags), list(orders)], lambda x: np.multiply(*[np.power(x[lag_k:-(lag_max-lag_k),k], order_k) for k, (lag_k, order_k) in enumerate(zip(lags, orders))])[:N_samples])
        else:
            yield ([list(lags), list(orders)], lambda x: np.zeros(x.shape[0]-lag_max + 1))

def bilinear_term(idxs):
    return lambda x: np.multiply(*[np.power(x[lag_k,k], order_k) for k, (lag_k, order_k) in enumerate(zip(*idxs))])

def project(q, y):
    return y.T @ q / (q.T @ q)

def sum_of_projections(pk, q_mat):
    pk = pk.reshape((-1,1))
    proj_sum = np.zeros((pk.shape[0]))

    if q_mat.ndim < 2:
        q_mat = q_mat.reshape((-1,1))
    for r in range(q_mat.shape[1]):
        proj_sum += pk.T @ q_mat[:,r]/(q_mat[:,r].T @ q_mat[:,r]) *q_mat[:,r]
    return proj_sum


def FROLS_volterra(X_input, y, rho, param=[10,10], f_max=100):
    X_input = np.reshape(X_input, (len(X_input),-1))
    y = np.reshape(y, (len(y), -1))
    X = X_input
    lag_max = param[0]
    N_samples = y.shape[0] - lag_max
    y = y[:N_samples]
    sigma = y.T @ y
    Ny = y.shape[1]
    Nx = X_input.shape[1]
    s_max = 10
    A = np.zeros((s_max, s_max))
    g_m = np.zeros((s_max, f_max))
    q_s = np.zeros((N_samples, s_max))
    q_m = np.zeros((N_samples, f_max))
    p_s = np.zeros((N_samples, s_max))
    g = np.zeros(s_max)
    g_s = np.zeros(f_max)

    ERRs = np.zeros(f_max)
    ERRs_max = np.zeros(s_max)
    ells = np.zeros(s_max, dtype=int)
    A[0,0] = 1
    f_idxs = []
    fun_list = islice(bilinear_term_generator(param, Nx, N_samples), f_max)
    for i, (f_idx, f) in enumerate(fun_list):
        f_idxs.append(f_idx)
        y_hat = f(X)
        q_m[:,i] = y_hat
        if np.allclose(y_hat, 0):
            g_m[0,i] = 0
        else:
            g_m[0,i] = y.T @ y_hat / (y_hat.T @ y_hat)
        ERRs[i] = g_m[0,i]**2 * y_hat.T @ y_hat/sigma

    ERRs_max[0] = max(ERRs)
    ells[0] = np.argwhere(ERRs == ERRs_max[0])[0][0]
    g[0] = g_m[0,ells[0]]
    g_s[0] = g[0]
    q_s[:,0] = q_m[:,ells[0]]
    p_s[:,0] = q_s[:,0]

    removed_fun_inds = [ells[0]]
    chosen_f_idxs = [f_idxs[ells[0]]]

    s = 1
    ESR = 10
    while (ESR >= rho) and (s < s_max):
        fun_list = islice(bilinear_term_generator(param, Nx, N_samples), f_max)

        g_m = np.zeros(f_max)
        q_m = np.zeros((N_samples, f_max))
        p_m = np.zeros((N_samples, f_max))
        ERRs = np.zeros(f_max)

        f_idxs = []

        for i, (f_idx,f) in enumerate(fun_list):
            if i not in removed_fun_inds:
                p_m[:,i] = f(X)[:N_samples]

                q_m[:,i] = p_m[:,i] - sum_of_projections(p_m[:,i:i+1], q_s[:,0:s])
                if np.alltrue(q_m[:,i] == 0):
                    g_m[i] = 0
                else:
                    g_m[i] = y.T @ q_m[:,i]/(q_m[:,i].T @ q_m[:,i])
                ERRs[i] = g_m[i]**2*(q_m[:,i].T @ q_m[:,i])/sigma
                if np.isnan(ERRs[i]):
                    a = 1

            else:
                ERRs[i] = 0
                _ = f(X)
            f_idxs.append(f_idx)


        ERRs_max[s] = max(ERRs)
        ell = np.argwhere(ERRs == ERRs_max[s])[0][0]

        q_s[:, s] = q_m[:, ell]
        g_s[s] = g_m[ell]
        p_s[:, s] = p_m[:, ell]

        ells[s] = ell

        chosen_f_idxs.append(f_idxs[ell])

        ESR = 1 - sum(ERRs_max)

        for i, qs in enumerate(q_s[:,:s].T):
            A[i, s] = project(qs, p_m[:,ell])
        A[s,s] = 1
        s+=1


    q = q_s[:, :s];
    g = g_s[:s];
    p = p_s[:, : s];
    A = A[:s, : s];
    ells = ells[:s];
    ERRs_max = ERRs_max[1:s - 1];



    print('FROLS Complete')
    fit_terms = [['x[%i]' % k + '^%i' %o + '(k-%i)' %l if o != 0 for o, l in zip(*f_idx)] for k, f_idx in enumerate(chosen_f_idxs)]
    print('Resulting fit: y = ')

    return q, g, ERRs_max, ells, A, p, chosen_f_idxs


