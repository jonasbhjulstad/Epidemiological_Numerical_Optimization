from casadi import *
from tqdm import tqdm
import numpy as np
from RK4.Integrator import RK4_Integrator
from numpy import matmul as mul
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
if __name__ == '__main__':

    from Parameters.Parameters_Vaccination_Flat import *
    Q = 0
    X0 = MX.sym('X0', 3,1)
    X = X0
    for j in range(M):
        k1, k1_q = f(X, u)
        k2, k2_q = f(X + DT / 2 * k1, u)
        k3, k3_q = f(X + DT / 2 * k2, u)
        k4, k4_q = f(X + DT * k3, u)
        X += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q += DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    F = Function('F', [X0, u], [X, Q])

    J = 0
    Xk = x0
    inc = 0
    X_plot = []
    U = MX.sym('U', N)
    for i in range(N):
        Xk, Qk = F(Xk, U[i])
        X_plot.append(Xk)
        J+= Qk

    X_plot = horzcat(*X_plot)

    h = vertcat(u_min - U, U - u_max)
    Nh = h.shape[0]
    grad_Phi = jacobian(J, U)

    s = MX.sym('s', Nh)
    mu = MX.sym('mu', Nh)
    tau = MX.sym('tau')

    w = vertcat(U,mu,s)

    grad_lag = grad_Phi.T + jacobian(h, U).T @ mu

    r = vertcat(grad_lag, h + s, mu*s - tau)

    Fr = Function('r',[w, tau], [r])
    Fr_obj = Function('r_obj', [tau], [r])
    Nr = Fr.sx_out(0).shape[0]
    Nw = w.shape[0]
    Nu = U.shape[0]
    delta_w = 1
    delta_c = 1e-3
    posdef_deltamat = DM.zeros(Nr, Nr)
    posdef_deltamat[0:Nu, 0:Nu] = DM.eye(Nu)*delta_w
    posdef_deltamat[Nu:, Nu:] = -DM.eye(Nr-Nu)*delta_c



    jac_Fr = Function('jac_Fr', [w, tau], [jacobian(r, w)])
    tol_lim = 1e-3
    U0 = np.array([[u0]*U.shape[0]])
    tau = 1
    mu0 = np.ones((1,Nh))
    s0 = mu0
    w0 = np.concatenate([U0, mu0, s0], axis=1).T

    G = rootfinder('root_r', 'newton', Fr)

    tau_tol = 1e-5
    wk = w0

    tau_list = [tau]
    iter_max = 7
    iter = 0

    u_plot = Function('u_plot', [w], [U])
    x_plot = Function('X_plot', [w], [X_plot])
    mu_plot = Function('h_plot', [w], [mu])
    s_plot = Function('h_plot', [w], [s])
    separate_w = Function('separate_w', [w], [U, mu, s])
    mu_s_vec = Function('mu_s', [w], [vertcat(mu, s)])
    wk_list = [wk]
    tol = 1e-3

    s_mu_g = -vertcat(s, mu)
    s_mu_lb = np.full((1, Nh), -inf)
    s_mu_ub = np.zeros((1,Nh))
    tau_b = 1-1e-5

    def max_step(x, d):
        alpha = 1
        condition = DM([1])
        while not condition.is_zero():
            alpha = ((1-tau_b)*x - x)
            condition = (x + alpha * d) < ((1 - tau_b) * x)
            alpha*=0.9
        return alpha

    max_iter = 100
    wk_diff = 1000
    wk_diff_list = []
    wk_diff_tol = 1e-4
    tau = 1
    while (wk_diff > wk_diff_tol) or (tau > 0.1):
        print(tau, wk_diff)
        error_k = tol + 1
        iter = 0
        wk_old = wk
        error = tol + 1
        while (error > tol) and (iter < max_iter):
            fk = Fr(wk, tau)
            jFr = jac_Fr(wk, tau)
            # for i in range(Nr):
            #     if jFr[i, i] < 0:
            #         jFr[i,i] = 1e3
            # d = -la.inv(jFr) @ fk
            # d_x, d_mu, d_s = separate_w(d)
            # xk, mu_k, s_k = separate_w(wk)
            #
            # # alpha_s = max_step(s_k, d_s)
            # # alpha_mu = max_step(mu_k, d_mu)
            # alpha_s = 1
            # alpha_mu = 1
            #
            # xk = xk + alpha_s*d_x
            # s_k = s_k + alpha_s*d_s
            # mu_k = mu_k + alpha_mu*d_mu


            # wk = vertcat(xk, s_k, mu_k)
            wk = wk - la.inv(jac_Fr(wk, tau)) @ Fr(wk, tau)
            error = norm_1(fk)
        wk_diff = norm_1(wk_old-wk)
        wk_diff_list.append(wk_diff)
        tau *=.9
        if np.any(np.isnan(wk)):
            break
            #print('Alphas: ' + str(alpha_s) + ', ' + str(alpha_mu) + ', error: ' + str(error))
        wk_list.append(wk)


    U_sols = [u_plot(wk) for wk in wk_list]
    X_sols = [x_plot(wk) for wk in wk_list]
    mu_sols = [mu_plot(wk) for wk in wk_list]
    s_sols = [s_plot(wk) for wk in wk_list]
    Q = 0


    import matplotlib.pyplot as plt
    from matplotlib import cm
    colormap = cm.get_cmap('Greys', len(X_sols))
    colors = colormap(np.linspace(.1, .8, len(X_sols)))

    tgrid = np.linspace(0,T,N)
    fig, ax = plt.subplots(4)
    for X_sol, U_sol, color in zip(X_sols[:-1], U_sols[:-1], colors[:-1]):
        ax[0].plot(tgrid.T, X_sol[0,:].T, color=color)
        ax[1].plot(tgrid.T, X_sol[1,:].T, color=color)
        ax[2].plot(tgrid.T, X_sol[2,:].T, color=color)

        ax[3].step(tgrid.T, U_sol, color=color)

    ax[0].plot(tgrid.T, X_sols[-1][0, :].T, color='k', marker='')
    ax[1].plot(tgrid.T, X_sols[-1][1, :].T, color='k', marker='')
    ax[2].plot(tgrid.T, X_sols[-1][2, :].T, color='k', marker='')

    ax[3].step(tgrid.T, U_sols[-1], color='k', marker='')
    ax[3].set_ylim([u_min, u_max])
    _ = [x.grid() for x in ax]

    ax[0].set_title('RK4 Single-Shooting N = %i, ' % N + "M = %i" % M + ", iterations = %i" %len(X_sols))

    ax[0].set_ylabel('S')
    ax[1].set_ylabel('I')
    ax[2].set_ylabel('R')
    ax[3].set_ylabel('u')
    ax[3].set_xlabel('time[days]')

    fig2, ax2 = plt.subplots()
    ax2.plot(wk_diff_list, color='k')
    ax2.grid()
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(r'$|\Delta w_k|$')

    plt.show()
    save = True
    if save:
        fig.savefig('../Figures/Symbolic_IPOPT_Traj.eps', format='eps')
        fig2.savefig('../Figures/Symbolic_IPOPT_error.eps', format='eps')






