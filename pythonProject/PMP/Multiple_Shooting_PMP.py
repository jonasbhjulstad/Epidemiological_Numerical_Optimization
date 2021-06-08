# %%

from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from RK4.Integrator import RK4_M_times_plot, integrator_N_times_plot
from Line_Search.Newton import armaijo_newton, newton_rhapson
from Custom_Logging.Iteration_logging import setup_custom_logger


def Multiple_shooting_PMP(param, is_armaijo=True):
    logger = setup_custom_logger('Multiple_Shooting_PMP_' + param)

    if param == 'Social Distancing':
        from Parameters.Parameters_Social_Distancing import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0
    lbd = MX.sym('lbd', (3,1))
    H = L + lbd.T @ xdot
    s = vertcat(x, lbd)
    s_dot = vertcat(xdot, -jacobian(H, x).T)

    # Functions for minimizing hamiltonian wrt. u
    grad_H_u = Function('grad_H_u', [u, s], [jacobian(H, u)])
    if is_armaijo:
        Gu = lambda u0, s: armaijo_newton(grad_H_u, Function('Fu', [u], [jacobian(grad_H_u, u)]), u0)
    else:
        Gu = rootfinder('Gu', 'newton', grad_H_u)
    def argmin_u(u0, s):
        u_opt = Gu(u0, s).full()[0][0]
        if u_opt < u_min:
            u_opt = u_min
        elif u_opt > u_max:
            u_opt = u_max
        return u_opt

    F = Function('f', [s, u], [s_dot, L])
    fk, Sk_plot, Qk_plot = RK4_M_times_plot(F, M, h, nx=2 * nx)
    f, S_plot, Q_plot = integrator_N_times_plot(fk, N, Sk_plot, Qk_plot, nx=2 * nx)

    U = MX.sym('U', N)
    S = MX.sym('S', (2 * nx, N+1))

    # Constraint setup

    # X0 = x0:
    g = [S[:nx, 0] - x0]
    for i in range(N):
        g.append(S[:, i + 1] - fk(S[:, i], U[i])[0])
    # Lambda_f = 0:
    g.append(S[nx:, -1])
    g = vertcat(*g)

    # Initial Conditions

    lbd0 = [1, 3, 5]
    s0 = [*x0, *lbd0]
    S0 = s0*(N+1)
    Sk = S0
    tol = 1e-3

    max_iter = 100

    S = S.reshape((-1,1))
    Fr = Function('Fr', [S, U], [g])
    jac_Fr = Function('Fr', [S, U], [jacobian(g, S)])

    S_sols = []
    U_sols = []
    error = tol + 1
    iter = 0
    while error > tol and iter < max_iter:

        S_sols.append(Sk)
        iter += 1
        # Calculate optimal U:
        U = []
        for i in range(N):
            sk = Sk[i:i+2*nx]
            U.append(argmin_u(u_max, sk))
        U_sols.append(U)
        # Solve newton-iteration:
        fk = Fr(Sk, U)
        Sk = Sk - la.inv(jac_Fr(Sk, U)) @ fk
        # assert not np.any(np.isnan(Sk)), 'PMP newton iteration returned nan'

        # Lambda_f stop criterion:
        error = norm_1(fk)
        logger.debug(error)

    logger.debug('PMP Solved, iterations: %i' %iter)
    logger.debug(S_sols)



if __name__ == '__main__':
    Multiple_shooting_PMP('Social Distancing')
