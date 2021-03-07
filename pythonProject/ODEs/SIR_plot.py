from ODEs.SIR import SIR, SIR_linearized
from RK4.Integrator import RK4_Integrator
import numpy as np
import matplotlib.pyplot as plt
from Parameters.Parameters_Vaccination_Flat import *
from tqdm import tqdm
if __name__ == '__main__':

    N_iter = 100000
    tspan = np.linspace(0,T, N_iter+1)
    DT = np.diff(tspan)
    beta = R0*alpha



    Xk = x0
    X = np.zeros((3, N_iter+1))
    X[:,0] = x0
    
    for k in range(N_iter):
        Xk, _ = f(Xk, u_max)
        X[:, k] = Xk.full()[:,0]

    fig, ax = plt.subplots(3)

    # ax[0].plot(tspan, S)
    # ax[0].plot(tspan, S_J)
    # ax[0].set_ylim([0,N])

    _ = [x.plot(tspan, j, color='k', label=name) for x, j, name in zip(ax, X, ['S', 'I', 'R'])]


    # ax.plot(tspan, I_J, '--', color='k', label='Linearization')
    # ax.plot(np.full(Nx, t_cross), np.linspace(0, N, Nx), '-.', 'k', label='I = N')
    # ax.set_ylim([0,N])
    # ax.set_xlim([0,tspan[-1]])
    ax[0].set_title('RK4 N = 10000, $\mathscr{R}_0$ = 6.5')
    ax[-1].set_xlabel('Time [days]')
    plt.grid()
    plt.legend()


    plt.show()
