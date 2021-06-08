import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle as pck
def Trajectory_Plot(SimDataName):
    with open(parent + '/data/' + SimDataName, 'rb') as f:
        SimData = pck.load(f)

    N_iter = len(SimData['U'])
    colormap = cm.get_cmap('Greys', N_iter)
    colors = colormap(np.linspace(.3,.8,N_iter))
    t = SimData['t']
    fig, ax = plt.subplots(4)
    for Uk, Xk, color in zip(SimData['U'], SimData['X'], colors):
        ax[0].plot(t, Xk[::3][:-2], color=color)
        ax[1].plot(t, Xk[1::3][:-2], color=color)
        ax[2].plot(t, Xk[2::3][:-2], color=color)
        ax[3].plot(t[::SimData['M']], Uk, color=color)   
    _ = [x.grid() for x in ax]
    plt.show()


if __name__ == '__main__':
    Trajectory_Plot('Single_Shooting_IPOPT_Social_Distancing.pck')
