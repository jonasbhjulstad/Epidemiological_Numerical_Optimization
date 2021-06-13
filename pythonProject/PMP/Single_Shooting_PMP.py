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
from Parameters.ODE_initial import N_pop, N, M, tgrid, tgrid_M
import pickle as pck


def Multiple_shooting_PMP(param, is_armaijo=False):

    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import u_min, u_max, L, xdot, x, nx, u, M, h, N, x0, u0, f
    lbd = MX.sym('lbd', (3,1))
    H = L + lbd.T @ xdot
    s = vertcat(x, lbd)
    s_dot = vertcat(xdot, -jacobian(H, x).T)

    # Functions for minimizing hamiltonian wrt. u
    H_u = Function('H_u', [u, s], [H])
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
    fsk, Sk_plot, Qk_plot = RK4_M_times_plot(F, M, h, nx=2 * nx)
    fs, S_plot, Q_plot = integrator_N_times_plot(fsk, N, Sk_plot, Qk_plot, nx=2 * nx)
    Fr = Function('Fr', [lbd, ])

    U = MX.sym('U', N)
    S = MX.sym('S', (2 * nx, N+1))

    # Constraint setup

    # X0 = x0:
    g = [S[:nx, 0] - x0]
    for i in range(N):
        g.append(S[:, i + 1] - fsk(S[:, i], U[i])[0])
    # Lambda_f = 0:
    g.append(S[nx:, -1])
    g = vertcat(*g)

    # Initial Conditions

    lbd0 = [1e-10, 1e-12, 1e-11]
    traj_initial = True
    S0 = [x0, lbd0]
    X0 = f(x0, u0)[0].full().squeeze()[:-nx]
    diff_scaler = [N_pop, N_pop, N_pop, 1, 1, 1]*(N+1)
    for i in range(N):
        if traj_initial:
            S0.append(X0[i:i+nx])
        else:
            S0.append(x0)
        S0.append(lbd0)
    S0 = np.concatenate(S0)
    trajinit= '_initial'
    Sk = S0
    diff_tol = 1e-6
    Sk_diff_norm = diff_tol+1

    max_iter = 100

    S = S.reshape((-1,1))
    Fr = Function('Fr', [S, U], [g])
    jac_Fr = Function('Fr', [S, U], [jacobian(g, S)])
    
    S_sols = []
    U_sols = []
    iter = 0
    uvec = np.linspace(0.5, u_max, 100)
    while (iter < max_iter) and (Sk_diff_norm > diff_tol):
        print('Iteration =  {}, Sk_diff = {}'.format(iter, Sk_diff_norm))
        iter += 1
        # Calculate optimal U:
        U = []
        for i in range(N):
            sk = Sk[i:i+2*nx]
            U.append(argmin_u(u0, sk))

            plt.plot(uvec, [H_u(u,sk).full().squeeze() for u in uvec])

        # print(U)
        # Solve newton-iteration:
        Sk_old = Sk
        plt.show()
        S_sol = (newton_rhapson(lambda s: Fr(s, U), lambda s: jac_Fr(s, U), Sk, tol=1e-6))
        Sk = S_sol[-1]
        # Sk_diff_norm = norm_1(np.divide(Sk-Sk_old, diff_scaler))
        Sk_diff_norm = norm_1(Sk-Sk_old)
        U_sols.extend([U]*len(S_sol))
        S_sols.extend(S_sol)




    X_sols_raw = [s.reshape((2*nx,-1))[:nx, :] for s in S_sols]
    lbd_sols_raw = [s.reshape((2*nx,-1))[nx:, :] for s in S_sols]
    S_sols_reconstructed = [S_plot(s[:2*nx], u) for s, u in zip(S_sols, U_sols)]
    X_sols = [s.reshape((2*nx, -1))[:nx,:-nx] for s in S_sols_reconstructed]
    lbd_sols = [s.reshape((2*nx, -1))[nx:,:-nx] for s in S_sols_reconstructed]

    Q_sols = [Q_plot(s_sol[:2*nx], u) for s_sol, u in zip(S_sols, U_sols)]

    a = 1
    sim_data = {'U': U_sols, 'lam_x': lbd_sols_raw,
                'X': X_sols,
                'Q': Q_sols,
                'X_raw': X_sols_raw,
                't_M': tgrid_M, 't': tgrid, 'N': N, 'M': M, 'f': sum(Q_sols[-1].full()), 'f_sols': [sum(q.full()) for q in Q_sols]}
    
    fname = 'Multiple_Shooting_PMP' + '_' + param + trajinit
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)

    return fname


if __name__ == '__main__':
    Multiple_shooting_PMP('Social_Distancing')
