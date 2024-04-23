from casadi import *
import numpy as np
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Line_Search.Newton import armaijo_newton, newton_rhapson
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd
import pickle as pck
from Parameters.ODE_initial import *

def Primal_Dual_Single_Shooting(param):
    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f, X_plot, Q_plot
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f, X_plot, Q_plot
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f, X_plot, Q_plot
    
    U = MX.sym('U', N)
    X, Q = f(x0, U)
    h = vertcat(u_min - U, U - u_max)
    Nh = h.shape[0]
    s = MX.sym('s', Nh)
    mu = MX.sym('mu', Nh)
    tau = MX.sym('tau')
    w = vertcat(U, mu, s)


    grad_Phi = jacobian(Q, U)

    grad_lag = grad_Phi.T + jacobian(h, U).T @ mu
    r = vertcat(grad_lag, h + s, mu * s - tau)

    Fr = Function('r', [w, tau], [r])
    jac_Fr = Function('jac_Fr', [w, tau], [jacobian(r, w)])


    tau = 1
    U0 = [u0]*N
    mu0 = [1]*Nh
    s0 = [tau/m for m in mu0]
    w0 = np.concatenate([U0, mu0, s0]).T

    G = rootfinder('root_r', 'newton', Fr)

    wk = w0
    tol = .5
    max_iter = 100
    wk_diff = 1000
    wk_diff_list = []
    wk_list = [w0]

    wk_opt = [w0]
    while (wk_diff > diff_tol) or (tau > tau_tol):
        print('tau = {}, norm_1(delta_w) = {}'.format(tau, wk_diff))
        wk_old = wk_opt[-1]
        f = lambda x: Fr(x, tau)
        grad_f = lambda x: jac_Fr(x, tau)


        wk_list.extend(newton_rhapson(f, grad_f, wk_old, tol=tol, max_iter=max_iter))
        wk_opt.append(wk_list[-1].full())
        wk_diff = norm_1(wk_old - wk_list[-1])
        wk_diff_list.append(wk_diff)
        tau *= tau_factor
        wk_list.append(wk)

    U_sols = [np.array(w[:N]) for w in wk_opt]
    X_sols = [X_plot(x0, u).full() for u in U_sols]
    Q_sols = [Q_plot(x0, u).full() for u in U_sols]
    mu_sols = [w[N:N+Nh] for w in wk_opt]
    s_sols = [w[N+Nh:N+2*Nh] for w in wk_opt]
    

    sim_data = {'U': U_sols, 'lam_x': mu_sols, 's': s_sols,
            'X': X_sols,
            'Q': Q_sols,
            'f_sols': [sum(q) for q in Q_sols],
            'f': sum(Q_sols[-1]),
            't_M': tgrid_M, 'N': N, 'M': M}
    
    fname = 'Single_Shooting_Primal_Dual_' + param
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)
    return fname

if __name__ == '__main__':
    Primal_Dual_Single_Shooting('Social_Distancing')


