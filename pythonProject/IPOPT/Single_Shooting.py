from casadi import *
import numpy as np
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Line_Search.Newton import armaijo_newton
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def Primal_Dual_Single_Shooting(param, tau_factor=.6):
    from Parameters.Parameters_Social_Distancing import f, x0, u_min, u_max, N
    method = 'Primal_Dual_Single_Shooting'

    U = MX.sym('U', N)

    X, Q = f(x0, U)

    h = vertcat(u_min - U, U - u_max)
    Nh = h.shape[0]
    grad_Phi = jacobian(Q, U)

    s = MX.sym('s', Nh)
    mu = MX.sym('mu', Nh)
    tau = MX.sym('tau')

    w = vertcat(U, mu, s)

    grad_lag = grad_Phi.T + jacobian(h, U).T @ mu
    r = vertcat(grad_lag, h + s, mu * s - tau)

    Fr = Function('r', [w, tau], [r])
    jac_Fr = Function('jac_Fr', [w, tau], [jacobian(r, w)])

    tau = 1
    U0 = [u_max]*N
    mu0 = [1]*Nh
    s0 = mu0
    w0 = np.concatenate([U0, mu0, s0]).T

    G = rootfinder('root_r', 'newton', Fr)

    wk = w0
    tol = .5
    max_iter = 100
    wk_diff = 1000
    wk_diff_list = []
    wk_diff_tol = 10
    tau = 1
    tau_tol = 1e-3
    beta = .8
    alpha = 1.0
    is_armaijo = False
    wk_list = [w0]

    logging.debug('Primal Dual SingleShooting Initialized')
    logging.debug('w0: ')
    logging.debug(w0)

    while (wk_diff > wk_diff_tol) or (tau > tau_tol):

        wk_old = wk_list[-1]
        f = lambda x: Fr(x, tau)
        grad_f = lambda x: jac_Fr(x, tau)

        wk_list.extend(armaijo_newton(f, grad_f, wk, tol=tol, max_iter=max_iter))

        wk_diff = norm_1(wk_old - wk_list[-1])
        wk_diff_list.append(wk_diff)
        tau *= tau_factor
        wk_list.append(wk)



if __name__ == '__main__':
    Primal_Dual_Single_Shooting('Social Distancing')


