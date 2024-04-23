import sys
from os.path import dirname, abspath
from Plot_Dirpath import PLOT_DIRECTORY

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pck
import matplotlib

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
    os.makedirs(PLOT_DIRECTORY + '/Trajectories/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Trajectories/' + SimDataName + '.pdf')

def Multiple_Shooting_Trajectory_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.9,N_iter))
    t = SimData['t']
    fig, ax = plt.subplots(4)
    for Uk, Xk, color in zip(SimData['U'], SimData['X_raw'], colors):
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
    ax[0].set_title('RK4 M = {}, N = {}, Iterations = {}'.format(SimData['M'], len(SimData['U'][0]), N_iter))
    os.makedirs(PLOT_DIRECTORY + '/Trajectories_Multiple_Shooting/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Trajectories_Multiple_Shooting/' + SimDataName + '.pdf')

#Single-Shooting
def Bounds_Objective_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    fig, ax = plt.subplots(2)
    for Qk,lam_x_k, color in zip(SimData['Q'], SimData['lam_x'], colors):
        ax[0].plot(t[::int(len(t)/len(lam_x_k))], lam_x_k, color=color)
        ax[1].plot(t, Qk[:-1], color=color)
    _ = [x.grid() for x in ax]
    ax[0].set_xlabel(r'$\lambda_u$')
    ax[1].set_xlabel(r'Q(objective)')
    os.makedirs(PLOT_DIRECTORY + '/Bounds_Objectives/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Bounds_Objectives/' + SimDataName + '.pdf')

#Multiple-Shooting
def Constraint_Objective_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t_M']
    fig, ax = plt.subplots(4)
    yscales = [max([max(np.abs(gk[6+i::3])) for gk in SimData['lam_g'][int(N_iter/2):]]) for i in range(3)]

    for Qk,lam_g_k, color in zip(SimData['Q'], SimData['lam_g'], colors):
        g0, g1, g2 = lam_g_k[::3], lam_g_k[1::3], lam_g_k[2::3]
        gStep = int(len(t)/len(g0))
        ax[0].plot(t[::gStep][:len(g0)], g0, color=color)
        ax[1].plot(t[::gStep][:len(g0)], g1, color=color)
        ax[2].plot(t[::gStep][:len(g0)], g2, color=color)
        ax[3].plot(t, Qk[:-1], color=color)
    _  = [x.set_ylim([-ys, ys]) for ys, x in zip(yscales, ax[:-1])]
    _ = [x.grid() for x in ax]
    ax[0].set_title('Multipliers and objective values')
    ax[0].set_ylabel('$\lambda_0$')
    ax[1].set_ylabel('$\lambda_1$')
    ax[2].set_ylabel('$\lambda_2$')
    ax[3].set_ylabel('Q (objective)')
    ax[-1].set_xlabel('Time[Days]')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    os.makedirs(PLOT_DIRECTORY + '/Constraint_Objectives/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Constraint_Objectives/' + SimDataName + '.pdf')
 

def Iterate_Objective_Constraint_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)
    N_iter = len(SimData['U'])
    fig, ax = plt.subplots(3)
    lab = lambda i: '$\lambda_' + str(i) + '$'
    [ax[0].plot([sum(abs(lam_g_k[i::3])) for lam_g_k in SimData['lam_g']], color='k', marker=m, label=lab(i)) for i, m in enumerate(['','o','x'])]
    ax[0].legend()
    ax[1].plot(sum(abs(lam_x_k) for lam_x_k in SimData['lam_x']), color='k')
    _ = [x.grid() for x in ax]
    _ = [x.set_yscale('log') for x in ax[:-1]]
    ax[2].plot(SimData['f_sols'], color='k')
    ax[0].set_ylabel('$\sum |\lambda|$')
    ax[1].set_ylabel('$\sum |\mu|$')
    ax[2].set_ylabel('f (objective)')
    ax[0].set_title('Magnitudes of objectives and constraints')
    ax[2].set_xlabel('Iteration')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    os.makedirs(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/' + SimDataName + '.pdf')

def Iterate_Objective_Constraint_Plot_No_Bounds(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)
    N_iter = len(SimData['U'])
    fig, ax = plt.subplots(2)
    lab = lambda i: '$\lambda_' + str(i) + '$'
    [ax[0].plot([sum(abs(lam_g_k[i::3])) for lam_g_k in SimData['lam_g']], color='k', marker=m, label=lab(i)) for i, m in enumerate(['','o','x'])]
    ax[0].legend()
    _ = [x.grid() for x in ax]
    _ = [x.set_yscale('log') for x in ax[:-1]]
    ax[1].plot(SimData['f_sols'], color='k')
    ax[0].set_ylabel('$\sum |\lambda|$')
    ax[1].set_ylabel('f (objective)')
    ax[0].set_title('Magnitudes of objectives and constraints')
    ax[1].set_xlabel('Iteration')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    os.makedirs(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/' + SimDataName + '.pdf')


#Single-Shooting
def Iterate_Objective_Bounds_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName + '.pck', 'rb') as f:
        SimData = pck.load(f)
    N_iter = len(SimData['U'])
    fig, ax = plt.subplots(2)
    lab = lambda i: '$\lambda_' + str(i) + '$'
    ax[0].plot([np.sum(np.sum(lam_x_k)) for lam_x_k in SimData['lam_x']], color='k')
    _ = [x.grid() for x in ax]
    ax[0].set_yscale('log')
    ax[1].plot(SimData['f_sols'], color='k')
    ax[0].set_ylabel('$\sum |\mu|$')
    ax[1].set_ylabel('f (objective)')
    ax[0].set_title('Magnitudes of objectives and constraints')
    ax[1].set_xlabel('Iteration')
    _ = [x.set_xticklabels('') for x in ax[:-1]]
    os.makedirs(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/', exist_ok=True)
    fig.savefig(PLOT_DIRECTORY + '/Iterate_Objective_Constraints/' + SimDataName + '.pdf')




if __name__ == '__main__':
    # Trajectory_Plot('Single_Shooting_SQP_Social_Distancing')
    # Constraint_Objective_Plot('Multiple_Shooting_SQP_Social_Distancing')
    # Iterate_Objective_Constraint_Plot('Multiple_Shooting_SQP_Social_Distancing')
    # Multiple_Shooting_Trajectory_Plot('Multiple_Shooting_SQP_Social_Distancing')
    # Trajectory_Plot('Single_Shooting_SQP_Isolation')
    # Iterate_Objective_Bounds_Plot('Single_Shooting_SQP_Isolation')
    Constraint_Objective_Plot('Multiple_Shooting_IPOPT_Social_Distancing_initial')
    plt.show()