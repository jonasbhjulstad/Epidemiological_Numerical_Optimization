from Integrators.Variational_SIR_Sensitivity import Variational_SIR_Sensitivity
from Integrators.Autodiff_SIR_Sensitivity import Autodiff_SIR_Sensitivity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
if __name__ == '__main__':

    dts = np.logspace(-2,0,6)
    dts = dts[::-1]
    dfs = []
    fig = plt.figure()
    outer_grid = gridspec.GridSpec(4,1,wspace=0.4, hspace=0.4)
    inner = []
    for k in range(3):
        inner.append(gridspec.GridSpecFromSubplotSpec(1, 3,
                                               subplot_spec=outer_grid[k], wspace=0.3, hspace=0.1))
    axs = []
    [axs.append(plt.Subplot(fig, inner[k][j])) for k in range(3) for j in range(3)]
    axs.append(plt.Subplot(fig, outer_grid[3]))


    colormap = cm.get_cmap('Greys', len(dts))
    colors = colormap(np.linspace(.2,1,len(dts)))
    plt.style.use('seaborn')
    for i,dt in enumerate(dts):
        #dfs.append(Variational_SIR_Sensitivity(dt, stop=18))
        dfs.append(Autodiff_SIR_Sensitivity(dt, stop=18))
        df = dfs[-1]
        t = df.index.values

        axs[0].plot(t, df['S'].values,color=colors[i])


        axs[1].plot(t, df['I'].values,color=colors[i])



        axs[2].plot(t, df['R'].values,color=colors[i])

        A = df['A'].values
        B = df['B'].values
        S_sense = np.reshape([k.T[0] for k in A], (-1,3))
        I_sense = np.reshape([k.T[1] for k in A], (-1,3))
        R_sense = np.reshape([k.T[2] for k in A],(-1,3))
        for k in range(3):

            axs[3+k].plot(t, S_sense[:,k], color=colors[i])

            axs[6+k].plot(t, I_sense[:,k], color=colors[i])

        u_sense = [[k[0], k[1]] for k in B]

        axs[-1].plot(t, u_sense, color=colors[i])

    [ax.set_xlabel('t') for ax in axs]
    [ax.set_ylabel('Individuals') for ax in axs[0:3]]
    [ax.set_ylabel(r'$\frac{dS}{dx}$') for ax in axs[3:6]]
    [ax.set_ylabel(r'$\frac{dI}{dx}$') for ax in axs[6:9]]
    axs[9].set_ylabel(r'$\frac{du}{dx}$')

    [ax.grid() for ax in axs]


    axs[1].set_title('State Trajectories')
    axs[4].set_title('Susceptible Sensitivities')
    axs[7].set_title('Infected Sensitivities')
    axs[9].set_title('Beta Sensitivity')
    [fig.add_subplot(ax) for ax in axs]
    plt.show()

    #Post-calc

    S = df['S'].values
    I = df['I'].values
    u = df['u'].values
    a = S*I
    ind = np.argmax(a)
    # axs[1].set_title('Sensitivities for S')
    # axs[2].set_title('Sensitivities for I')

# plt.savefig('../Figures/Variational_Sensitivities.eps', format='eps')