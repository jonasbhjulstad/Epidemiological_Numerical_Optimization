import numpy as np
from ODEs.SIR import SIR
from scipy import integrate
import pandas as pd
def dF_dx(x, u, alpha):
    return np.array([[-u * x[1], -u * x[0], 0],
                     [u * x[1], u * x[0] - alpha, 0],
                     [0, alpha, 0]])


def dF_du(x):
    return np.array([-x[0] * x[1],
                     x[0] * x[1],
                     0])

def A_ODE(x, u, alpha):
    A =x[3:12].reshape(3,3)
    Adot = np.matmul(dF_dx(x[0:3], u, alpha), A)
    return Adot.reshape(-1)

def B_ODE(x, u, alpha):
    B = x[12:21]
    Bdot = np.matmul(dF_dx(x[0:3], u, alpha), B) + dF_du(x[0:3])
    return Bdot

def F_tot(t, x, param):
    alpha = param[0]
    u = param[1]
    print("ODE")
    return np.concatenate((SIR(x, alpha, u), A_ODE(x,u,alpha), B_ODE(x,u,alpha)))

if __name__ == '__main__':
    dt = 1e-3
    X0 = np.array([997, 3, 0])
    N = np.sum(X0)
    alpha = 1.0/3.0
    R0 = 10
    u = R0*alpha/N

    A0 = np.eye(3).reshape(-1)
    B0 = np.array([0,0,0])
    t = [0]
    t_end = 1.0
    state = [np.concatenate((X0,A0,B0))]
    solver = integrate.ode(F_tot).set_integrator("dopri5")#, rtol=1e-1, atol=1e-1)
    solver.set_f_params([alpha, u])
    solver.set_initial_value(state[0], t[0])
    perc = 0
    print("Solving..")
    while solver.successful() and solver.t < t_end:
        tk = t[-1] + dt
        if(tk > 0.1*(t_end-t[0])):
            perc = perc+10
            print('%d%%', perc)

        t.append(tk)
        state.append(solver.integrate(tk))

        S = state[0]
        I = state[1]
        R = state[2]



        fig, axs = plt.subplots(4, 1)
        axs[0].plot(t, S, label="S")
        axs[0].plot(t, I, label="I")
        axs[0].plot(t, R, label = "R")
        axs[0].legend()

        axs[1].plot(t, S_sense)
        axs[1].legend(['S wrt. S', 'I wrt. I', 'I wrt. R'])
        axs[1].set_title('Sensitivities for S')

        axs[2].plot(t, I_sense)
        axs[2].legend(['I wrt. S', 'I wrt. I', 'I wrt. R'])
        axs[2].set_title('Sensitivities for I')

        u_sense = [[k[0], k[1]] for k in B]


        axs[3].plot(t, u_sense)
        axs[3].legend(['S wrt u', 'I wrt u'])

    plt.show()

    At = pd.Series(A, index=t)
    Bt = pd.Series(B, index=t)
    St = pd.Series(S, index=t)
    It = pd.Series(I, index=t)
    Rt = pd.Series(R, index=t)

    d = {'A': At, 'B': Bt, 'S': St, 'I': It, 'R': Rt}

    df = pd.DataFrame(d, index=t)
    if save:
        df.to_pickle('../data/Variational_sense.pck')

