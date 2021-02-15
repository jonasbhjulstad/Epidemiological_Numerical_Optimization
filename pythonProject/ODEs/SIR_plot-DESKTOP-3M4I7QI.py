from ODEs.SIR import SIR, SIR_linearized
from RK4.Integrator import RK4_Integrator
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':

    T_end = 40
    tspan = np.linspace(0,T_end, 10000)
    DT = np.diff(tspan)
    alpha = 0.1
    R0 = 10
    beta = R0*alpha

    N = 10000

    I0 = 100
    S0 = N-I0
    param = [alpha, R0, N]
    X0 = [N-I0, I0, 0]
    def SIR_linearized(x, U):
        global S0
        global I0
        J = np.array([[-beta*I0/N, -beta*S0/N, 0],
                      [beta*I0/N, beta*S0/N-alpha, 0],
                      [0, alpha, 0]])
        return J @ x

    ode = lambda x, R0: SIR(x, alpha, N, R0)
    X = [X0]
    X_J = [X0]
    for dt in DT:
        X.append(RK4_Integrator(ode, X[-1], R0, dt))
        X_J.append(RK4_Integrator(SIR_linearized, X_J[-1], R0, dt))
        # if X_J[-1][1] > N:
        #     break
    S = [x[0] for x in X]
    I = [x[1] for x in X]
    R = [x[2] for x in X]

    S_J = [x[0] for x in X_J]
    I_J = [x[1] for x in X_J]
    R_J = [x[2] for x in X_J]

    fig, ax = plt.subplots(3)

    Nx = len(X)
    tspan = tspan[:Nx]
    ax[0].set_title('Susceptible')
    ax[0].plot(tspan, S)
    ax[0].plot(tspan, S_J)
    ax[0].set_ylim([0,N])

    ax[1].set_title('Infected')

    ax[1].plot(tspan, I)
    ax[1].plot(tspan, I_J)
    ax[1].set_ylim([0,N])

    ax[2].set_title('Recovered')
    ax[2].plot(tspan, R)
    ax[2].plot(tspan, R_J)
    ax[2].set_ylim([0,N])

    plt.show()
