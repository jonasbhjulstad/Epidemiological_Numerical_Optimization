from casadi import *
import numpy as np
from RK4.Integrator import RK4_Integrator
from numpy import matmul as mul
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == '__main__':
    nu = 1
    nx = 3
    NX = 10
    NU = 5
    NU_step = nx*NX/NU+1
    N_repeat_U = int(NX/NU)
    nw = nu*NU + nx*NX
    M = 1000
    DT = 0.001
    U = MX.sym('U', NU)
    U_dupe = vertcat(*[repmat(U[i], N_repeat_U, 1) for i in range(U.shape[0])])

    u_min = 1e-4
    u_max = 10
    alpha = 1.0/3.0


    def SIR(x,beta):
        S = x[0]
        I = x[1]
        R = x[2]

        S_dot = -beta * S * I
        I_dot = beta * S * I - alpha * I
        R_dot = alpha * I
        return vertcat(S_dot, I_dot, R_dot)


    Wu = 0.01
    x = MX.sym('x', 1,nx)
    u = MX.sym('u', 1)
    F = Function('F', [x,u], [SIR(x,u), x[1]**2 + Wu/(u**2)])

    Q = 0
    x0 = MX.sym('x0', 3,1)
    xk = x0
    u = MX.sym('u', 1)
    for j in range(M):
        k1, k1_q = F(xk, u)
        k2, k2_q = F(xk + DT / 2 * k1, u)
        k3, k3_q = F(xk + DT / 2 * k2, u)
        k4, k4_q = F(xk + DT * k3, u)
        X = xk + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

    fx = Function('fx', [x0, u], [X])
    X = MX.sym('X', NX,3)
    fq = Function('fq', [x0, u], [Q])
    fx_N = fx.mapaccum('fx_N',NX)
    fq_N = fq.map(NX)
    Q_list = fq_N(fx_N(x0, U_dupe), U_dupe)
    Q_tot = 0
    for i in range(Q_list.shape[0]):
        Q_tot += Q_list[i]

    Q = Function('Q', [x0, U], [Q_tot])
    X = MX.sym('X',NX, nx)

    w = vertcat(x0, X.reshape((-1,1)), U)

    g = Function('g', [w], [(fx_N(x0, U_dupe) - X.T).reshape((-1,1))])


    grad_Phi = jacobian(Q(x0, U), w)

    h = Function('h', [w], [vertcat(u_min-U, U-u_max)])

    Nh = h.numel_out()
    mu = MX.sym('mu', Nh)
    Ng = g.numel_out()
    lbd = MX.sym('lbd', Ng)

    W = vertcat(w,lbd,mu)


    F = Function('F', [x0, u], [SIR(x0, u)])
    F_N = F.mapaccum(NX)

    grad_lag = Function('grad_lag', [x0, U, lbd], [Q_list + lbd.T @ F_N(x0, U_dupe).reshape((-1,1))])
    H_tot = 0
    H_list = grad_lag(x0, U, lbd)
    for i in range(H_list.shape[0]):
        H_tot += H_list[i]

    W = vertcat(*[x0, U, lbd])

    H = Function('H', [W], [H_tot])


    x0 = MX([1000,200,0])
    lbd0 = MX(np.full((Ng, 1), 0.5))

    H_fixed = Function('H_fixed', [U], [H(vertcat(x0, U, lbd0))])

    grad_H = Function('grad_H', [U], [jacobian(H_fixed(U), U)])
    grad_2H = Function('grad_2H', [U], [jacobian(grad_H(U), U)])

    uk = MX(np.full((NU, 1), 0.1))
    tol = 1e-3
    fu = 1+tol
    while fu > tol:
        fu = grad_H(uk)
        fu_dot = grad_2H(uk)
        uk = uk - fu @ la.inv(fu_dot)























