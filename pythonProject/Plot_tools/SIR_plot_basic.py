from Parameters.Parameters_Vaccination_Flat import *
import matplotlib.pyplot as plt
import numpy as np
from RK4.Integrator import RK4_Integrator
from ODEs.SIR import SIR

if __name__ == '__main__':

    f = lambda x, u: SIR(x, alpha, N_pop, u)
    N_iter = 10000

    X =[x0]
    tgrid = np.linspace(0, 365, N_iter + 1)
    DT = np.diff(tgrid)
    for i, dt in zip(range(N_iter), DT):
        X.append(RK4_Integrator(f, X[-1], R0, dt))

    fig, ax = plt.subplots(3)

    _ = [x.plot(tgrid, j, color='k') for x, j in zip(ax, np.array(X).T)]
    _ = [x.grid() for x in ax]

    ax[0].set_title('RK4-Integration N = %i' %N_iter + ', $\mathscr{R}_0 = %.1f$' %R0 + ', $I_0$ = %i' %x0[1])
    ax[-1].set_xlabel('time [days]')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    _ = [x.set_ylabel(s) for x, s in zip(ax, ['S', 'I', 'R'])]
    plt.show()

    fig.savefig('../Figures/Uncontrolled_SIR.eps', format = 'eps')