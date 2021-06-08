import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
plt.style.use('seaborn-white')
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Collocation.collocation_coeffs import collocation_coeffs
from Lagrange_Polynomials.Polynomial_plot import Collocation_Trajectory
import numpy as np
import pandas as pd
import time
def Objective(x, u):
    Wu = 0.01
    return x**2 + Wu/(u**2)

def ObjectivePlot(f, x_bound, y_bound, iter, res):
    x = np.linspace(x_bound[0], x_bound[1], res)
    y = np.linspace(y_bound[0], y_bound[1], res)
    z = 0
    X,Y = np.meshgrid(x,y)
    Z = f(X, Y)
    levels = np.linspace(0.1,0.8,2)
    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contourf(X, Y, Z, 100,levels=levels, cmap=cm.coolwarm)
    plt.show()
    return fig, ax
class ObjectivePlotter(object):
    def __init__(self, f, df, res=5, I_bound=[], u_bound = []):
        self.f = f
        self.x_bound = I_bound
        self.u_bound = u_bound
        if I_bound == []:
            self.x_bound = [0,df['N'][0]]
        if u_bound == []:
            self.u_bound = [min(min(df['u'])),max(max(df['u']))]
        self.res = res
        self.fig, self.ax = ObjectivePlot(f, self.x_bound, self.u_bound,0,  res)
        self.df = df
        self.step = df.index[1]-df.index[0]
        self.i_iter = 0
    def add_x_constraints(self, con, res=100):
        x = np.linspace(self.x_bound[0], self.x_bound[1],res)
        y = map(con, x)
        self.ax.plot(x,y)

    def add_y_constraints(self, con, res=100):
        y = np.linspace(self.u_bound[0], self.u_bound[1],res)
        x = map(con, y)
        self.ax.plot(x,y)

    def traj_plot(self, iter):
        I = self.df['I'].values
        u = self.df['u'].values
        self.ax.plot(I,u)
    def iter_plot(self, iter):
        ax = self.ax

        t = self.df['t'][iter][:-1]
        I = self.df['I'][iter][:-1]
        u = self.df['u'][iter]
        ax.plot(t, I,u)


        t0 = self.df['t'][iter][0]
        tend = self.df['t'][iter][-1]
        toff = tend*0.2
        ax.plot(t,u,self.x_bound[0], zdir='y', linestyle='dashed', color='g')
        ax.plot(t,I,self.u_bound[0], zdir='z', linestyle='dashed', color='r')

        ax.set_xlim(t0-toff, tend)
        ax.set_ylim(self.x_bound[0], self.x_bound[1])
        ax.set_zlim(self.u_bound[0], self.u_bound[1])

        ax.set_xlabel('t')
        ax.set_ylabel('I')
        ax.set_zlabel('u')


        plt.show()

    def theta_iter_lines(self, iter):
        d = 3
        tau_root = [0, 0.15505102572168222, 0.6449489742783179, 1.0]
        ax = self.ax
        I_thetas = self.df['I_thetas'][iter]
        u = self.df['u'][iter]
        t = self.df['t'][iter][:-1]
        dt = t[1]-t[0]

        t_poly = np.linspace(0,1,15)

        lines = []
        for k in range(I_thetas.shape[0]/(d+1)):
            theta_k = I_thetas[4*k:4*k+4]
            s = Collocation_Trajectory(d, theta_k,tau_root)
            lines.append(ax.plot(t_poly + t[k], s(t_poly), [u[k]]*t_poly.shape[0], color='k', linewidth=1))
        return lines
    def interactive_button_plot(self, step):
        self.i_iter = 0
        def onclick(event):
            self.ax.clear()
            self.theta_iter_plot(self.i_iter)
            self.i_iter += step

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        plt.draw()
        # N_steps = df.index[-1]/steps
        # self.fig.canvas.draw()




if __name__ == '__main__':
    plt.close('all')
    df = pd.read_pickle('../data/Singleshoot_data.pck')

    res = 100
    # fig,ax = ObjectivePlot(Objective, x_bound, u_bound,0, 5)


    OP = ObjectivePlotter(Objective, df)
    # OP.theta_iter_lines(530)
    OP.iter_plot(12)
    # OP.interactive_plot(20)