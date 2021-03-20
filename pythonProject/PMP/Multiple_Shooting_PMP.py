# %%

from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%

nx = 3
nu = 1



def RK4_Integrator(f, X, U, DT):
    k1 = f(X, U)
    k2 = f(X + DT / 2 * k1, U)
    k3 = f(X + DT / 2 * k2, U)
    k4 = f(X + DT * k3, U)
    X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X



def Multiple_shooting_PMP(param):

    if param == 'Social Distancing':
        from Parameters.Parameters_Social_Distancing import u_min, u_max, Wu, x0, M, DT, T, N_pop,s_dot, N, M, alpha, grad_h_u, u_min, u_max, x, lbd
        u_sol = lambda x, lbd: u_sol_social_distancing(x, lbd)
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import u_min, u_max, Wu, x0, M, DT, T, N_pop, s_dot, N, M, grad_h_u, u_min, u_max, x, lbd
        u_sol = lambda x, lbd: u_sol_vaccination(x, lbd)

    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import u_min, u_max, Wu, x0, M, DT, T, N_pop, s_dot, xdot, N, M, grad_h_u, u_min, u_max, x, lbd
        u_sol = lambda x, lbd: u_sol_isolation(x, lbd)

    def argmin_H_vacc(x, lbd):
        u_opt = (lbd[2]-lbd[0])*x[0]/(2*Wu)
        if u_opt < u_min:
            return u_min
        elif u_opt > u_max:
            return u_max
        else:
            return u_opt

    x0_num = x0
    u = MX.sym('u')
    x0 = MX.sym('x0', nx)
    lbd0 = MX.sym('lbd0', nx)

    s0 = vertcat(x0, lbd0)
    sk = s0
    s_plot = [s0]
    for i in range(M):
        sk = RK4_Integrator(s_dot, sk, u, DT)
        s_plot.append(sk)
    f = Function('f', [s0, u], [sk])
    s_plot = Function('s_plot', [s0, u], [vertcat(*s_plot)])
    # %%

    # "Lift" initial conditions
    X0 = MX.sym('x0', nx)
    lbd_0 = MX.sym('lbd_0', nx)
    x0 = x0_num[:nx]
    Sk = vertcat(X0, lbd_0)

    w = [Sk]
    g = [X0 - x0]
    U = []
    U0 = []
    u0 = u_max
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        U += [Uk]
        U0 += [u0]

        # Integrate till the end of the interval
        Sk_end = f(Sk, Uk)

        # New NLP variable for state at end of interval
        if k < (N - 1):
            Sk = MX.sym('S_' + str(k + 1), 2 * nx)
            w += [Sk]
            # Add equality constraint
            g += [Sk_end - Sk]

    g += [Sk[nx:]]

    g = vertcat(*g)
    w = vertcat(*w)
    U = vertcat(*U)

    f_r = Function('f_r', [w, U], [g])
    grad_r = Function('grad_r', [w, U], [jacobian(f_r(w, U), w)])

    Sk = DM(np.concatenate([x0_num, [1, 2, 0.001]]))
    w0 = repmat(Sk, N)
    tol = 1e-3
    cond = tol + 1
    lam_tol = 1e-3
    lam_f = DM([1, 1, 1])
    U_sols = []


    while norm_1(lam_f) > lam_tol:
        U = []
        Sk = w0[:2*nx]
        for k in range(N):
            U.append(argmin_H_vacc(Sk[:nx], Sk[nx:2*nx]))
            Sk = f(Sk, U[-1])
        tol = 1e-3
        wk = w0
        U = vertcat(*U)
        wk_sols = [w0]
        err = tol + 1
        errs = [err]
        while err > tol:
            fk = f_r(wk, U)
            wk = wk - la.inv(grad_r(wk, U)) @ fk
            err = norm_1(fk)
            errs.append(err)
            print(err)
        U_sols.append(U)
        wk_sols.append(wk)
        Sk = wk[:Sk.shape[0]]
        lam_f = wk[-nx:]
        w_sol = wk_sols[-1]

    a = grad_r(w0, U)
    print("grad_r_rank = ", la.matrix_rank(a))
    x_sols = [[w_sol[0::nx * 2], w_sol[1::nx * 2], w_sol[2::nx * 2]] for w_sol in wk_sols]
    lbd_sols = [[w_sol[3::nx * 2], w_sol[4::nx * 2], w_sol[5::nx * 2]] for w_sol in wk_sols]

    s_plot_sol = []
    for i in range(N):
        s_plot_sol.append(s_plot(w_sol[6*i:2*nx+6*i], U[i]))

    s_plot_sol = vertcat(*s_plot_sol).reshape((6,-1))
    # %%


    tgrid = np.linspace(0, T, N)
    import matplotlib.pyplot as plt
    from matplotlib import cm

    colormap = cm.get_cmap('Greys', len(x_sols))
    colors = colormap(np.linspace(.1, .8, len(x_sols)))

    fig1, ax1 = plt.subplots(4)
    fig2, ax2 = plt.subplots(3)
    # for x_sol, lbd_sol, color in zip(x_sols, lbd_sols, colors):
    #     ax1[0].plot(tgrid, x_sol[0], color=color)
    #     ax1[1].plot(tgrid, x_sol[1], color=color)
    #     ax1[2].plot(tgrid, x_sol[2], color=color)
    #     ax1[3].step(tgrid, U, color='k')
    #
    #     ax2[0].plot(tgrid, lbd_sol[0], color=color)
    #     ax2[1].plot(tgrid, lbd_sol[1], color=color)
    #     ax2[2].plot(tgrid, lbd_sol[2], color=color)
    tgrid = np.linspace(0,T, s_plot_sol.shape[1])
    tgrid_u = np.linspace(0,T,U.shape[0])
    ax1[0].plot(tgrid, s_plot_sol[0,:].full().squeeze(), color='k')
    ax1[1].plot(tgrid, s_plot_sol[1,:].full().squeeze(), color='k')
    ax1[2].plot(tgrid, s_plot_sol[2,:].full().squeeze(), color='k')
    ax1[3].step(tgrid_u, U[:], color='k')

    ax2[0].plot(tgrid, s_plot_sol[3,:].full().squeeze(), color='k')
    ax2[1].plot(tgrid, s_plot_sol[4,:].full().squeeze(), color='k')
    ax2[2].plot(tgrid, s_plot_sol[5,:].full().squeeze(), color='k')


    _ = [x.grid() for x in np.concatenate([ax1, ax2])]
    ax1[0].set_title('RK4 Multiple-Shooting N = %i, ' % N + "M = %i" % M + ", iterations = %i" % len(x_sols))

    _ = [x.set_ylabel(l) for l, x in zip(['S', 'I', 'R', 'u'], ax1)]
    _ = [x.set_ylabel(l) for l, x in zip([r'$\lambda_0$', r'$\lambda_1$', r'$\lambda_2$'], ax2)]
    _ = ax1[-1].set_xlabel('time[days]')
    _ = ax2[-1].set_xlabel('time[days]')
    ax2[0].set_title('$\lambda$-Multipliers')
    plt.show()

if __name__ == '__main__':
    Multiple_shooting_PMP('Vaccination')