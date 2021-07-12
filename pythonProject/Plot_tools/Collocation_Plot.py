import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pck
import matplotlib

#Single-Shooting
def Iterate_Objective_Bounds_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)
    N_iter = len(SimData['U'])
    fig, ax = plt.subplots(2)
    lab = lambda i: '$\lambda_' + str(i) + '$'
    ax[0].plot(sum(np.abs(lam_x_k) for lam_x_k in SimData['lam_x']), color='k')
    _ = [x.grid() for x in ax]
    ax[0].set_yscale('log')
    ax[1].plot(SimData['f_sols'], color='k')
    ax[0].set_ylabel('$\sum |\lambda_x|$')
    ax[1].set_ylabel('f (objective)')
    ax[0].set_title('Magnitudes of objectives and constraints')
    ax[1].set_xlabel('Iteration')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    fig.savefig(parent + '/overleaf/Figures/Iterate_Objective_Constraints/' + SimDataName + '.pdf')
#Multiple-Shooting
def Direct_Collocation_Constraint_Objective_Plot_d1(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    d = SimData['d']
    fig, ax = plt.subplots(4)
    yscales = [max([max(np.abs(gk[6+i::6])) for gk in SimData['lam_g']]) for i in range(3)]
    for Qk,lam_g_k, color in zip(SimData['Q'], SimData['lam_g'], colors):
        g0, g1, g2, gth0, gth1, gth2 = [lam_g_k[6+i::6] for i in range(6)]
        gStep = int(len(t)/len(g0))
        ax[0].plot(t[::gStep], g0, color=color, label='x')
        # ax[0].plot(t[::gStep][1:], gth0, color=color, label=r'$\theta$')
        ax[1].plot(t[::gStep], g1, color=color)
        # ax[1].plot(t[::gStep][1:], gth1, color=color)
        ax[2].plot(t[::gStep], g2, color=color)
        # ax[2].plot(t[::gStep][1:], gth2, color=color)
        ax[3].plot(t[::gStep][1:], Qk[:-1], color=color)
    # _ = [x.set_yscale('log') for x in ax[:-1]]
    _ = [x.grid() for x in ax]
    # _ = [x.set_ylim([.01, -.01]) for x in ax[:-1]]
    ax[0].set_title('Multipliers and objective values')
    ax[0].set_ylabel('$\lambda_0$')
    ax[1].set_ylabel('$\lambda_1$')
    ax[2].set_ylabel('$\lambda_2$')
    ax[3].set_ylabel('Q (objective)')
    ax[-1].set_xlabel('Time[Days]')
    _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    fig.savefig(parent + '/overleaf/Figures/Collocation_Constraints/' + SimDataName + '.pdf')
 
def Direct_Collocation_Theta_Constraint_Plot_d1(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    d = SimData['d']
    fig, ax = plt.subplots(3)
    yscales = [max([max(np.abs(gk[9+i::6])) for gk in SimData['lam_g']]) for i in range(3)]
    for Qk,lam_g_k, color in zip(SimData['Q'], SimData['lam_g'], colors):
        g0, g1, g2, gth0, gth1, gth2 = [lam_g_k[6+i::6] for i in range(6)]
        gStep = int(len(t)/len(g0))
        # ax[0].plot(t[::gStep], g0, color=color, label='x')
        ax[0].plot(t[::gStep][1:], gth0, color=color)
        # ax[1].plot(t[::gStep], g1, color=color)
        ax[1].plot(t[::gStep][1:], gth1, color=color)
        # ax[2].plot(t[::gStep], g2, color=color)
        ax[2].plot(t[::gStep][1:], gth2, color=color)
        # ax[3].plot(t[::gStep][1:], Qk[:-1], color=color)
    _ = [x.grid() for x in ax]
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    ax[0].set_title('Collocation Constraint Multipliers')
    ax[0].set_ylabel(r'$\lambda_{\theta,0}$')
    ax[1].set_ylabel(r'$\lambda_{\theta,1}$')
    ax[2].set_ylabel(r'$\lambda_{\theta,2}$')
    ax[-1].set_xlabel('Time[Days]')
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    fig.savefig(parent + '/overleaf/Figures/Collocation_Constraints/' + SimDataName + '_theta.pdf')
 
def Direct_Collocation_Bounds_Plot_d1(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    d = SimData['d']
    fig, ax = plt.subplots(3)
    yscales = [max([max(np.abs(gk[9+i::6])) for gk in SimData['lam_x']]) for i in range(3)]
    for Qk,lam_x_k, color in zip(SimData['Q'], SimData['lam_x'], colors):
        xth0, xth1, xth2 = [lam_x_k[3+i::7] for i in range(3)]
        gStep = int(len(t)/len(xth0))
        # ax[0].plot(t[::gStep], g0, color=color, label='x')
        ax[0].plot(t[::gStep], xth0, color=color)
        # ax[1].plot(t[::gStep], g1, color=c
        # olor)
        ax[1].plot(t[::gStep], xth1, color=color)
        # ax[2].plot(t[::gStep], g2, color=color)
        ax[2].plot(t[::gStep], xth2, color=color)
        # ax[3].plot(t[::gStep][1:], Qk[:-1], color=color)
    _ = [x.grid() for x in ax]
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    ax[0].set_title('Collocation Bounds Multipliers')
    ax[0].set_ylabel(r'$\mu_0$')
    ax[1].set_ylabel(r'$\mu_1$')
    ax[2].set_ylabel(r'$\mu_2$')
    ax[-1].set_xlabel('Time[Days]')
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    fig.savefig(parent + '/overleaf/Figures/Collocation_Bounds/' + SimDataName + '.pdf')
 

def Direct_Collocation_Theta_Bounds_Plot_d1(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    d = SimData['d']
    fig, ax = plt.subplots(3)
    yscales = [max([max(np.abs(gk[9+i::6])) for gk in SimData['lam_x']]) for i in range(3)]
    for Qk,lam_x_k, color in zip(SimData['Q'], SimData['lam_x'], colors):
        xth0, xth1, xth2 = [lam_x_k[6+i::7] for i in range(3)]
        gStep = int(len(t)/len(xth0))
        # ax[0].plot(t[::gStep], g0, color=color, label='x')
        ax[0].plot(t[::gStep], xth0, color=color)
        # ax[1].plot(t[::gStep], g1, color=c
        # olor)
        ax[1].plot(t[::gStep], xth1, color=color)
        # ax[2].plot(t[::gStep], g2, color=color)
        ax[2].plot(t[::gStep], xth2, color=color)
        # ax[3].plot(t[::gStep][1:], Qk[:-1], color=color)
    _ = [x.grid() for x in ax]
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    ax[0].set_title('Collocation Bounds Multipliers')
    ax[0].set_ylabel(r'$\mu_{\theta,0}$')
    ax[1].set_ylabel(r'$\mu_{\theta,1}$')
    ax[2].set_ylabel(r'$\mu_{\theta,2}$')
    ax[-1].set_xlabel('Time[Days]')
    # _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    fig.savefig(parent + '/overleaf/Figures/Collocation_Bounds/' + SimDataName + '_theta.pdf')
 

def Direct_Collocation_Trajectory_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.9,N_iter))
    t = SimData['t']
    fig, ax = plt.subplots(4)
    for Uk, Xk, color in zip(SimData['U'], SimData['X'], colors):
        ax[0].plot(t, np.reshape(Xk[0,:len(t)], -1), color=color)
        ax[1].plot(t, np.reshape(Xk[1,:len(t)], -1), color=color)
        ax[2].plot(t, np.reshape(Xk[2,:len(t)], -1), color=color)
        ax[3].plot(t[::int(len(t)/len(Uk))][:-1], Uk[:-1], color=color)   
    _ = [x.grid() for x in ax]
    ax[0].set_ylabel('S')
    ax[1].set_ylabel('I')
    ax[2].set_ylabel('R')
    ax[3].set_ylabel('u')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    ax[3].set_xlabel('Time [days]')
    ax[0].set_title('Direct Collocation d = {}, N = {}, Iterations = {}'.format(SimData['d'], SimData['N'], N_iter))
    fig.savefig(parent + '/overleaf/Figures/Trajectories_Multiple_Shooting/' + SimDataName + '.pdf')
def Iter_Objective_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.9,N_iter))
    t = SimData['t']
    fig, ax = plt.subplots(3)

    lab = lambda i: '$\lambda_' + str(i) + '$'
    ax[0].plot([sum(abs(lam_g_k)) for lam_g_k in SimData['lam_g']], color='k')
    ax[1].plot([sum(abs(lam_x_k)) for lam_x_k in SimData['lam_x']], color='k')


    ax[-1].plot(SimData['f_sols'], color='k')
    _ = [x.grid() for x in ax]
    ax[-1].set_ylabel('f (objective)')
    ax[0].set_ylabel(r'$\sum |\lambda|)$')
    ax[1].set_ylabel(r'$\sum |\mu|$')
    # _ = [x.set_yscale('log') for x in ax[:-1]]
    ax[-1].set_xlabel('Iteration')
    ax[0].set_title('Magnitudes of Objectives and Constraints')

    fig.savefig(parent + '/overleaf/Figures/Iterate_Objective_Constraints/' + SimDataName + '.pdf')

if __name__ == '__main__':
    # Direct_Collocation_Trajectory_Plot('Direct_Collocation_IPOPT_'+param+'_initial')
    # Direct_Collocation_Constraint_Objective_Plot_d1('Direct_Collocation_IPOPT_'+param+'_initial')
    # # Direct_Collocation_Constraint_Objective_Plot('Direct_Collocation_IPOPT_' + param + '_initial')
    # Direct_Collocation_Theta_Constraint_Plot_d1('Direct_Collocation_IPOPT_' + param + '_initial')
    # Direct_Collocation_Theta_Bounds_Plot_d1('Direct_Collocation_IPOPT_' + param + '_initial')
    # Direct_Collocation_Bounds_Plot_d1('Direct_Collocation_IPOPT_' + param + '_initial')
    [Iter_Objective_Plot('Direct_Collocation_IPOPT_'+param+'_initial') for param in ['Social_Distancing', 'Vaccination', 'Isolation']]