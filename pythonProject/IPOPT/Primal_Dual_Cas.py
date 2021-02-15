from casadi import *
from tqdm import tqdm
import numpy as np
from RK4.Integrator import RK4_Integrator
from numpy import matmul as mul
import numpy.linalg as la
import matplotlib.pyplot as plt

if __name__ == '__main__':

    T = 28.  # Time horizon
    N = 50  # number of control intervals
    M = 30
    DT = T/N/M

    # Declare model variables
    S = MX.sym('S')
    I = MX.sym('I')
    R = MX.sym('R')
    x = vertcat(S, I, R)
    u = MX.sym('u')
    N_pop = 5.3e6
    u_min = 0.5
    u_max = 6.5
    Wu = N_pop ** 2 / (u_max - u_min) / 10

    alpha = 0.2
    beta = u * alpha
    I0 = 2000
    x0 = [N_pop - I0, I0, 0]
    # Model equations
    xdot = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I, alpha * I)
    # Objective term
    L = I ** 2 - Wu * u ** 2

    f = Function('f', [x,u], [xdot, L])
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
    jac_Fr = Function('jac_Fr', [w, tau], [jacobian(r, w)])
    tol_lim = 1e-3
    u0 = 1000
    U0 = np.array([[u0]*U.shape[0]])
    tau = 1
    mu0 = np.ones((1,Nh))
    s0 = mu0
    w0 = np.concatenate([U0, mu0, s0], axis=1)

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
    tau_b = 1-1e-8

    def max_step(x, d):
        alpha = 1
        condition = DM([1])
        while not condition.is_zero():
            condition = (x + alpha * d) < ((1 - tau_b) * x)
            alpha*=0.9
        return alpha/.9

    while tau > tau_tol:
        tau_list.append(tau)
        tau*=.9
    for tau in tqdm(tau_list):
        error = tol + 1
        while error > tol:
            fk = Fr(wk, tau)
            d = -la.inv(jac_Fr(wk, tau)) @ fk
            d_x, d_mu, d_s = separate_w(d)
            xk, mu_k, s_k = separate_w(wk)

            alpha_s = max_step(s_k, d_s)
            alpha_mu = max_step(mu_k, d_mu)

            xk = xk + alpha_s*d_x
            s_k = s_k + alpha_s*d_s
            mu_k = mu_k + alpha_mu*d_mu

            wk = vertcat(xk, s_k, mu_k)

            error = norm_1(fk)
            print('Alphas: ' + str(alpha_s) + ', ' + str(alpha_mu) + ', error: ' + str(error))
        wk_list.append(wk)


    U_sols = [u_plot(wk) for wk in wk_list]
    X_sols = [x_plot(wk) for wk in wk_list]
    mu_sols = [mu_plot(wk) for wk in wk_list]
    s_sols = [s_plot(wk) for wk in wk_list]
    Q = 0

    tgrid = np.linspace(0,T,N)
    fig, ax = plt.subplots(4)
    ax[0].plot(tgrid.T, X_sols[-1][0,:].T)
    ax[1].plot(tgrid.T, X_sols[-1][1,:].T)
    ax[2].plot(tgrid.T, X_sols[-1][2,:].T)

    ax[3].plot(tgrid.T, U_sols[-1])
    ax[3].set_ylim([u_min, u_max])

    plt.show()








