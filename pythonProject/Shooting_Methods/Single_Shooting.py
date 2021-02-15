#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
from casadi import *
from Callbacks.Singleshoot import Singleshoot_CB
import pandas as pd
from Parsing.IPOPT_Parse import parse_IPOPT_log
if __name__ == '__main__':
    T = 28. # Time horizon
    N = 50 # number of control intervals

    # Declare model variables
    S = MX.sym('S')
    I = MX.sym('I')
    R = MX.sym('R')
    x = vertcat(S, I, R)
    u = MX.sym('u')
    N_pop = 5.3e6
    u_min = 0.5
    u_max = 6.5
    Wu = N_pop**2/(u_max-u_min)/100

    alpha = 0.2
    beta = u*alpha
    I0 = 2000
    x0 = [N_pop - I0, I0, 0]
    # Model equations
    xdot = vertcat(-beta*S*I/N_pop, beta*S*I/N_pop-alpha*I, alpha*I)

    # Objective term
    L = I**2 - Wu*u**2

    # Formulate discrete time dynamics
    # Fixed step Runge-Kutta 4 integrator
    M = 30 # RK4 steps per interval
    DT = T/N/M
    f = Function('f', [x, u], [xdot, L])
    X0 = MX.sym('X0', 3)
    U = MX.sym('U')
    X = X0
    Q = 0
    X_plot = [X0]
    for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X+= DT/6*(k1 +2*k2 +2*k3 +k4)
       X_plot.append(X)
       Q += DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = Function('F', [X0, U], [X, Q])
    Fx_plot = Function('Fx_plot', [X0, U], [vertcat(*X_plot)])

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    u0 = u_max

    # Formulate the NLP
    Xk = x0
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [u_min]
        ubw += [u_max]
        w0 += [u0]

        # Integrate till the end of the interval
        xf, qf = F(Xk, Uk)
        Xk = xf
        J+= qf

    opts = {}
    opts["expand"] = True

    # IPOPT_CB = IPOPT_Callback()
    # opts["ipopt"]["iteration_callback"] = IPOPT_CB

    iter_step = 1
    # opts["ipopt"]["output_file"] = '../data/log.opt'
    # opts["ipopt"]["file_print_level"] = 0
    # opts["ipopt"]["print_frequency_iter"] = iter_step
    opts["calc_f"] = True
    opts["calc_g"] = True

    NV = N
    Ng = 0

    CB = Singleshoot_CB('Singleshoot_CB', NV, Ng,1,iter_step)
    opts["iteration_callback"] = CB
    opts["iteration_callback_step"] = iter_step
    opts["ipopt.tol"] = 1e-12



    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']

    u_opt = w_opt.full()
    N_sols = len(CB.x_sols)

    u_sols = CB.x_sols
    x_sols = []
    x_plots = []
    for i in range(N_sols):
        x_opt = [x0]
        x_plot = []
        for k in range(N):
            xf, qf = F(x_opt[k], u_sols[i][k])
            x_plot.append(Fx_plot(x_opt[k], u_sols[i][k]).full())
            x_opt += [xf.full()]
        x_plot = np.concatenate(x_plot)
        x1_opt = [r[0] for r in x_opt]
        x2_opt = [r[1] for r in x_opt]
        x3_opt = [r[2] for r in x_opt]

        x_plots.append([x_plot[0::3], x_plot[1::3], x_plot[2::3]])
        x_sols.append([x1_opt, x2_opt, x3_opt])

    tgrid = [T/N*k for k in range(N+1)]
    tgrid = np.linspace(0,21, N*M + N)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    colormap = cm.get_cmap('Greys', len(x_sols))
    colors = colormap(np.linspace(.1, .8, len(x_sols)))

    fig, axs = plt.subplots(4,1)

    tgrid_u = np.linspace(0,T,N)

    u = v = np.zeros((len(tgrid), len(u_sols[0])))
    for i, sol in enumerate(x_plots):
        axs[0].plot(tgrid, sol[0], '-', color = colors[i])
        axs[1].plot(tgrid, sol[1], '-', color = colors[i])
        axs[2].plot(tgrid, sol[2], '-', color = colors[i])
        axs[3].step(tgrid_u,  u_sols[i], '-.', color = colors[i])

    axs[0].plot(tgrid, sol[0], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[1].plot(tgrid, sol[1], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[2].plot(tgrid, sol[2], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[3].step(tgrid_u, u_sols[i], linestyle='',marker='o',markersize=2.5, color='k')

    axs[3].set_xlabel('t[days]')
    _ = [x.set_xticklabels([]) for x in axs[:-1]]

    axs[0].set_ylabel('S')
    axs[1].set_ylabel('I')
    axs[2].set_ylabel('R')
    axs[3].set_ylabel('R0')
    _ = [x.set_xticklabels([]) for x in axs[:-1]]

    _ = [x.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) for x in axs[:-1]]

    axs[0].set_title('RK4 Single-Shooting N = %i, ' %N+ "M = %i" %M)


    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()

    fig2, ax2 = plt.subplots(2)

    lam_x = np.array(CB.lam_x_sols).squeeze()
    [ax2[0].plot(tgrid_u, lam, color=color,marker='o', linestyle='', markersize=2.5) for i, (lam, color) in enumerate(zip(lam_x, colors))]
    ax2[1].plot(CB.f_sols, color='k', label='objective value')

    ax2[0].set_title(r'Multipliers for bounds on $U$')
    ax2[1].legend()
    ax2[0].set_xlabel('time')
    ax2[1].set_xlabel('Iteration')
    _ = [x.grid() for x in ax2]