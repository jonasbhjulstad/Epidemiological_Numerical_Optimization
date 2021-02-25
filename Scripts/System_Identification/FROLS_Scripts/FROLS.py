import numpy as np
from itertools import islice, tee, combinations, chain

def bilinear_term_generator(param):
    lag_max = param[0]
    order_max = param[1]
    lags = tee(combinations(range(lag_max), 2), 2)
    orders = tee(combinations(range(order_max), 2), 2)

    lag_order = [chain(lag, order) for lag, order in zip(lags, orders)]
    tot_combs = combinations(chain(*lag_order), 2)

    for (l0, o0), (l1, o1) in tot_combs:
        yield lambda x: np.multiply(np.power(x[l0:-(lag_max-l0),0], o0), np.power(x[l1:-(lag_max-l1),1], o1))

def bilinear_term(param):
    l0, o0, l1, o1 = param[0], param[1], param[2], param[3]
    return lambda x: np.multiply(np.power(x[l0:], o0), np.power(x[l1:], o1))

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


def FROLS(X_input, y, rho, generator_fun=bilinear_term_generator, param=[10,10], f_max=100):
    lag_max = param[0]
    N_samples = y.shape[0] - lag_max
    y = y[:N_samples]
    X = np.concatenate([X_input, y[N_samples:]], axis=1)
    sigma = y.T @ y
    Nx = X.shape[1]
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

    fun_list = islice(generator_fun(param), f_max)
    for i, f in enumerate(fun_list):
        y_hat = f(X)[:N_samples]
        q_m[:,i] = y_hat
        g_m[0,i] = y.T @ y_hat / (y_hat.T @ y_hat)
        ERRs[i] = g_m[0,i]**2 * y_hat.T @ y_hat/sigma

    ERRs_max[0] = max(ERRs)
    ells[0] = np.argwhere(ERRs == ERRs_max[0])[0][0]
    g[0] = g_m[0,ells[0]]
    g_s[0] = g[0]
    q_s[:,0] = q_m[:,ells[0]]
    p_s[:,0] = q_s[:,0]

    removed_fun_inds = [i]

    s = 1
    ESR = 10
    while (ESR >= rho) and (s < s_max):
        fun_list = islice(generator_fun(param), f_max)
        g_m = np.zeros(f_max)
        q_m = np.zeros((N_samples, f_max))
        p_m = np.zeros((N_samples, f_max))
        ERRs = np.zeros(f_max)

        for i, f in enumerate(fun_list):
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


        ERRs_max[s] = max(ERRs)
        ell = np.argwhere(ERRs == ERRs_max[s])[0][0]

        q_s[:, s] = q_m[:, ell]
        g_s[s] = g_m[ell]
        p_s[:, s] = p_m[:, ell]

        ells[s] = ell

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

    return q, g, ERRs_max, ells, A, p


