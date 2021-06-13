from casadi import *
import numpy as np
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Line_Search.Newton import armaijo_newton, newton_rhapson
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
import pickle as pck
from Parameters.ODE_initial import tgrid, tgrid_M, M


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def Primal_Dual_Multiple_Shooting(param, tau_factor=.6, traj_initial=True):
    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import N_pop,f, x0, u_min, u_max, N, fk, nx, Q_plot, X_plot
    if param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import N_pop,f, x0, u_min, u_max, N, fk, nx, Q_plot, X_plot
    if param == 'Isolation':
        from Parameters.Parameters_Isolation import N_pop,f, x0, u_min, u_max, N, fk, nx, Q_plot, X_plot
    method = 'Primal_Dual_Multiple_Shooting'

    U = MX.sym('U', N)
    X = MX.sym('X', (nx, N))

    #Constraint setup

    Q = 0
    g = [X[:, 0] - x0]
    Xk = x0
    for i in range(N - 1):
        Xk, Qk = fk(Xk, U[i])
        g.append(X[:, i + 1] - Xk)
        Q += Qk
    V = vertcat(U, X.reshape((-1, 1)))
    g = vertcat(*g)
    Ng = g.shape[0]

    h = vertcat(u_min - U, U - u_max)
    Nh = h.shape[0]
    grad_Phi = jacobian(Q, V)

    #Primal Dual variables

    s = MX.sym('s', Nh)
    lbd = MX.sym('lbd', Ng)
    mu = MX.sym('mu', Nh)
    tau = MX.sym('tau')

    w = vertcat(V, lbd, mu, s)

    #Primal Dual formulation

    grad_lag = grad_Phi.T + jacobian(g, V).T @ lbd + jacobian(h, V).T @ mu
    r = vertcat(grad_lag, g, h + s, mu * s - tau)

    Fr = Function('r', [w, tau], [r])
    jac_Fr = Function('jac_Fr', [w, tau], [jacobian(r, w)])

    #Initial Values

    tau = 1
    U0 = [0.001] * N
    if traj_initial:
        X0 = f(x0, U0)[0].full().squeeze()[:-nx]
    else:
        X0 = x0 * N

    lbd0 = np.linspace(0, 10, Ng)
    mu0 = [.5] * Nh
    s0 = mu0
    w0 = np.concatenate([U0, X0, lbd0, mu0, s0]).T

    wk = w0

    jac_Fr0 = jac_Fr(w0, tau)
    init_rank = la.matrix_rank(jac_Fr0)
    assert init_rank == jac_Fr0.shape[0], 'Initial rank deficiency: %i' % init_rank + ', should be %i' % jac_Fr0.shape[
        0]

    tol = 1e-3
    max_iter = 100
    wk_diff = 1000
    diff_scaler = [u_max-u_min]*N + [N_pop]*(N)*nx + [1]*(Ng + 2*Nh)
    wk_diff_list = []
    wk_diff_tol = 900
    tau = 1
    tau_tol = 1e-3
    is_armaijo = False
    wk_list = [w0]
    wk_opt_list = [w0]

    def root_fun(wk, tau):
        f = lambda w: Fr(w, tau)
        jac_f = lambda w: jac_Fr(w, tau)
        if is_armaijo:
            return armaijo_newton(f, jac_f, wk, tol=tol, max_iter=max_iter, verbose=True)
        else:
            return newton_rhapson(f, jac_f, wk, tol=tol, max_iter=max_iter, verbose=True)
    iter = 0
    wk_diff_norm = 1000
    while wk_diff_norm > wk_diff_tol or (tau > tau_tol) or (iter > max_iter):
        print('tau = {}, norm_1(delta_w) = {}'.format(tau, wk_diff_norm))
        wk_old = wk_list[-1]
        wk_list.extend(root_fun(wk, tau))
        wk_opt_list.append(wk_list[-1])
        wk_diff = wk_old - wk_list[-1]
        wk_diff_list.append(wk_diff)
        wk_diff_norm = norm_1(np.divide(wk_diff, diff_scaler))
        tau *= tau_factor
        wk_list.append(wk)
        iter+=1

    # Format solution data
    separate_w = Function('sep_w', [w], [U, X, lbd, mu, s])
    U_list = []
    X_list = []
    lbd_list = []
    mu_list = []
    s_list = []
    for w_sol in wk_opt_list:
        U_opt, X_opt, lbd_opt, mu_opt, s_opt = separate_w(w_sol)
        _ = [x.append(y.full()) for x, y in zip([U_list, X_list, lbd_list, mu_list, s_list],[U_opt, X_opt, lbd_opt, mu_opt, s_opt])]
    



    sim_data = {'U': U_list,'lam_g': lbd_opt, 'lam_x': mu_opt,
            'X': [X_plot(x0, u) for u in U_list],
            'Q': [Q_plot(x0, u) for u in U_list],
            'X_raw': X_list,
            't_M': tgrid_M, 't': tgrid, 'N': N, 'M': M}
    fname = parent + '/data/Multiple_Shooting_Primal_Dual_' + param
    with open(fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)
    
    return fname


if __name__ == '__main__':
    Primal_Dual_Multiple_Shooting('Social_Distancing', traj_initial=True)
