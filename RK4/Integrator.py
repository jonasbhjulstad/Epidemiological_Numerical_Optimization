from ODEs.SIR import SIR
from casadi import *


def RK4_Integrator(f, X, U, DT):
    k1, _ = f(X, U)
    k2, _ = f(X + DT / 2 * k1, U)
    k3, _ = f(X + DT / 2 * k2, U)
    k4, _ = f(X + DT * k3, U)
    X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def RK4_M_times(f, M, h, nx=3, nu=1):
    DT = h / M
    X0 = MX.sym('X', nx)
    X = X0
    u = MX.sym('U', nu)
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, u)
        k2, k2_q = f(X + DT / 2 * k1, u)
        k3, k3_q = f(X + DT / 2 * k2, u)
        k4, k4_q = f(X + DT * k3, u)
        X += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q += DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    F = Function('F', [X0, u], [X, Q])
    return F


def RK4_M_times_plot(F, M, h, nx=3, nu=1):
    DT = h / M
    X0 = MX.sym('X', nx)
    X = X0
    u = MX.sym('U', nu)
    Q = 0
    X_plot = []
    Q_plot = []
    for j in range(M):
        k1, k1_q = F(X, u)
        k2, k2_q = F(X + DT / 2 * k1, u)
        k3, k3_q = F(X + DT / 2 * k2, u)
        k4, k4_q = F(X + DT * k3, u)
        X += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Qk = DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        Q += Qk
        X_plot.append(X)
        Q_plot.append(Qk)
    f = Function('f', [X0, u], [X, Q])
    X_plot = Function('X_plot', [X0, u], [vertcat(*X_plot)])
    Q_plot = Function('Q_plot', [X0, u], [vertcat(*Q_plot)])
    return f, X_plot, Q_plot


def integrator_N_times_plot(fk, N, Xk_plot, Qk_plot, nx=3, nu=1):
    X0 = MX.sym('X', nx)
    Xk = X0
    U = MX.sym('U', N)
    X_plot = [X0]
    X = [X0]
    Q = 0
    Q_plot = []

    for i in range(N):
        X_plot.append(Xk_plot(Xk, U[i]))
        Q_plot.append(Qk_plot(Xk, U[i]))
        Xk, Qk = fk(Xk, U[i])
        X.append(Xk)
        Q+=Qk
    X_plot.append(Xk)
    Q_plot.append(Qk)

    X_plot = Function('X_plot', [X0, U], [vertcat(*X_plot)])
    Q_plot = Function('Q_plot', [X0, U], [vertcat(*Q_plot)])
    f = Function('f', [X0, U], [vertcat(*X), Q])
    return f, X_plot, Q_plot

