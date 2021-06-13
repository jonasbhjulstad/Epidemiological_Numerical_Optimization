import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class collocation_plot(object):
    def __init__(self, x_plot, u_plot, thetas, tgrid):
        self.tgrid = tgrid
        self.tgrid_u = tgrid[:-1]
        self.tf = tgrid[-1]
        self.N = len(tgrid)
        self.iter  = 0
        self.d = 3
        self.x_plot = x_plot
        self.u_plot = u_plot
        self.thetas = thetas
        self.N_iter = thetas.shape[0]
        self.tgrid_poly = []


    def solution_plot(self,x_opt, u_opt):
        tgrid = np.linspace(0, self.tf, self.N + 1)
        tgrid_u = np.linspace(0, self.tf, self.N)

        # Plot the results
        fig, ax = plt.subplots(2,1)

        ax[0].plot(tgrid, x_opt.T, '-.')
        ax[0].set_title("SIR Collocation Optimization")
        ax[1].ticklabel_format(useMathText=True)
        ax[1].step(tgrid_u, u_opt.T, '-')

        ax[0].set_xlabel('time')
        ax[0].legend(['S trajectory', 'I trajectory', 'R trajectory'])
        ax[1].legend(['u trajectory'])
        ax[0].grid()
        plt.show()
        return fig, ax

    def iteration_plot(self, iteration_step, poly_resolution=10, full_plot=False):
        poly_list, tau_root = collocation_polynomials(self.d)
        t_poly = np.linspace(0,1,poly_resolution)
        self.tgrid_poly = [t_poly*(self.tgrid[k+1]-self.tgrid[k]) + self.tgrid[k] for k in range(self.N-1)]
        tgrid_u = np.linspace(0, self.tf, self.N)
        poly_vals = np.array([p(t_poly) for p in poly_list])

        def calc_poly_traj(theta_k):
            S_traj = [theta_k[0,4*i:4*i+(self.d+1)] @ poly_vals for i in range(self.N)]
            I_traj = [theta_k[1,4*i:4*i+(self.d+1)] @ poly_vals for i in range(self.N)]
            R_traj = [theta_k[2,4*i:4*i+(self.d+1)] @ poly_vals for i in range(self.N)]

            return S_traj, I_traj, R_traj


        def theta_plot(theta_k, x_plot, u_plot, axs, iter, color='k', marker='', markersize=3):
            S_traj, I_traj, R_traj = calc_poly_traj(theta_k.values)
            [axs[0].plot(tk, s, color=color) for tk, s in zip(self.tgrid_poly, S_traj)]
            [axs[1].plot(tk, i, color=color) for tk, i in zip(self.tgrid_poly, I_traj)]
            [axs[2].plot(tk, r, color=color) for tk, r in zip(self.tgrid_poly, R_traj)]

            axs[3].step(tgrid_u, u_plot, color=color)
            if marker != '':
                [axs[0].plot([tk[0], tk[-1]], [s[0], s[-1]], color='k', marker=marker, markersize=markersize) for tk, s in zip(self.tgrid_poly, S_traj)]
                [axs[1].plot([tk[0], tk[-1]], [i[0], i[-1]], color='k', marker=marker, markersize=markersize) for tk, i in zip(self.tgrid_poly, I_traj)]
                [axs[2].plot([tk[0], tk[-1]], [r[0], r[-1]], color='k', marker=marker, markersize=markersize) for tk, r in zip(self.tgrid_poly, R_traj)]
                axs[3].step(tgrid_u, u_plot, color='k', marker=marker, markersize=2.5)



def Trajectory_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.9,N_iter))
    t = SimData['t_M']
    fig, ax = plt.subplots(4)
    for Uk, Xk, color in zip(SimData['U'], SimData['X'], colors):
        ax[0].plot(t, Xk[::3][:-2], color=color)
        ax[1].plot(t, Xk[1::3][:-2], color=color)
        ax[2].plot(t, Xk[2::3][:-2], color=color)
        ax[3].plot(t[::int(len(t)/len(Uk))], Uk, color=color)   
    _ = [x.grid() for x in ax]
    ax[0].set_ylabel('S')
    ax[1].set_ylabel('I')
    ax[2].set_ylabel('R')
    ax[3].set_ylabel('u')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    ax[3].set_xlabel('Time [days]')
    ax[0].set_title('RK4 M = {}, N = {}, Iterations = {}'.format(SimData['M'], len(SimData['U'][0]), N_iter))
