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
import logging
from Custom_Logging.Iteration_logging import setup_custom_logger


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def Primal_Dual_Multiple_Shooting(param, tau_factor=.6, traj_initial=True):
    if param == 'Social Distancing':
        from Parameters.Parameters_Social_Distancing import f, x0, u_min, u_max, N, fk, nx
    if param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import f, x0, u_min, u_max, N, fk, nx
    if param == 'Isolation':
        from Parameters.Parameters_Isolation import f, x0, u_min, u_max, N, fk, nx
    method = 'Primal_Dual_Multiple_Shooting'
    logger = setup_custom_logger(method + '_' + param)

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
    U0 = [u_max] * N
    if traj_initial:
        X0 = f(x0, [u_max] * N)[0].full().squeeze()[:-nx]
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

    tol = .5
    max_iter = 100
    wk_diff = 1000
    wk_diff_list = []
    wk_diff_tol = 10
    tau = 1
    tau_tol = 1e-3
    is_armaijo = True
    wk_list = [w0]

    logger.debug('Primal Dual Multiple Shooting Initialized')
    logger.debug('w0: ')
    logger.debug(w0)

    def root_fun(wk, tau):
        f = lambda w: Fr(w, tau)
        jac_f = lambda w: jac_Fr(w, tau)
        if is_armaijo:
            return armaijo_newton(f, jac_f, wk, tol=tol, max_iter=max_iter,
                                  logger=logger)
        else:
            return newton_rhapson(f, jac_f, wk, tol=tol, max_iter=max_iter,
                                  logger=logger)

    while (wk_diff > wk_diff_tol) or (tau > tau_tol):
        wk_old = wk_list[-1]
        wk_list.extend(root_fun(wk, tau))
        wk_diff = norm_1(wk_old - wk_list[-1])
        wk_diff_list.append(wk_diff)
        tau *= tau_factor
        wk_list.append(wk)


if __name__ == '__main__':
    Primal_Dual_Multiple_Shooting('Vaccination')
