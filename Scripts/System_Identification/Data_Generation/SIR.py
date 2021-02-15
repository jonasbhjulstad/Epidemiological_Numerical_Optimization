from ODE.Epidemiological import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
def RK4_Integrator(f, X, U, DT):
    k1 = f(X, U)
    k2 = f(X + DT / 2 * k1, U)
    k3 = f(X + DT / 2 * k2, U)
    k4 = f(X + DT * k3, U)
    X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X
def generate_model_Data(f,x0, param, U, t0=0.0, t_end=0.3, DT=1e-3, M=3):

    Nu = len(U)
    xk = x0
    X = [x0]
    t = [t0]
    f_fixed = lambda x, u: f(x, u,param)
    N_steps = int((t_end-t0)/DT)

    i_u =0
    for k in range(N_steps):
        if k%(N_steps/Nu) == 0:
            u = U[i_u]
            i_u+=1

        for i in range(M):
            xk = RK4_Integrator(f_fixed, xk, u, DT/M)
        X.append(xk)
        t.append(t[-1] + M*DT)
    return t, X

if __name__ == '__main__':
    alpha = 1.0/3.0

    x0 = [1000, 20, 0]
    U = [0.5, 0.1]

    param = SIRD_parameters()

    u = param['beta']

    t, X = generate_model_Data(SIRD, x0, param, U)
    plt.plot(t, X)
    plt.show()
    a = 1

    #SIRD Parameters
