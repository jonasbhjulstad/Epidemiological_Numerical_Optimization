# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from casadi import *
# from ODEs.SIR import SIR
import numpy as np
import matplotlib.pyplot as plt
from Collocation.collocation_coeffs import collocation_coeffs, collocation_polynomials
from Callbacks.CasADI_Collocation_Callback import DataCallback
from Plot_tools.Collocation_Plot import collocation_plot
import pandas as pd
import pickle as pck
import xarray as xr
# Press the green button in the gutter to run the script.




def SIR(X, R0):
    beta = R0*alpha
    return vertcat(-beta * X[0, :] * X[1, :]/N_pop, beta * X[0, :] * X[1, :]/N_pop - alpha * X[1, :], alpha * X[1, :])

if __name__ == '__main__':
    from Parameters.Parameters_Social_Distancing import *
    plt.close()

    d = 3
    B, C, D, tau_root = collocation_coeffs(d)

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    do_solve = True
    solve='casADI'
    data_path = r'../data/'
    # Time horizon
    nx = 3
    nu = 1
    tf = T
    S = MX.sym('S')
    I = MX.sym('I')
    R = MX.sym('R')
    x = vertcat(S, I, R)
    u = MX.sym('u')


    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    use_g_bounds = False
    xg_min = [0,0,0]
    xg_max = [N_pop, N_pop, N_pop]
    if use_g_bounds:
        x_min = [-inf] * 3
        x_max = [inf] * 3
    else:
        x_min = [0,0,0]
        x_max = [N_pop, N_pop, N_pop]


    u0 = u_max
    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = MX.sym('X0', 3)
    w.append(Xk)
    lbw.append(x0)
    ubw.append(x0)
    w0.append(x0)
    x_plot.append(Xk)
    x_thetas = [Xk]


    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w.append(Uk)
        lbw.append([u_min])
        ubw.append([u_max])
        w0.append([u0])
        u_plot.append(Uk)
        x_thetas.append(Xk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = MX.sym('X_' + str(k) + '_' + str(j), 3)
            Xc.append(Xkj)
            w.append(Xkj)
            x_thetas.append(Xkj)

            lbw.append(x_min)
            ubw.append(x_max)
            if use_g_bounds:
                g.append(Xkj)
                lbg.append(xg_min)
                ubg.append(xg_max)
            w0.append(x0)

        # Loop over collocation points
        Xk_end = D[0] * Xk
        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(d): xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j - 1], Uk)
            g.append(h * fj - xp)
            lbg.append([0, 0, 0])
            ubg.append([0, 0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1];

            # Add contribution to quadrature function
            J = J + B[j] * qj * h

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k + 1), 3)
        w.append(Xk)
        lbw.append(x_min)
        ubw.append(x_max)
        if use_g_bounds:
            g.append(Xk)
            lbg.append(xg_min)
            ubg.append(xg_max)
        w0.append(x0)
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end - Xk)
        lbg.append([0, 0, 0])
        ubg.append([0, 0, 0])

    # Concatenate vectors
    w = vertcat(*w)
    x_thetas = vertcat(*x_thetas)
    g = vertcat(*g)
    x_plot = horzcat(*x_plot)
    u_plot = horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    Ng = g.shape[0]
    NW = w.shape[0]
    v_opt = []

    if do_solve:
        # Create an NLP solver
        NU = N
        trajectories = Function('trajectories', [w], [x_plot, u_plot, x_thetas.reshape((3, -1))], ['w'],
                                ['x', 'u', 'thetas'])

        param = {'nx': NW, 'nu': nu, 'ng': Ng, 'd': d, 'tgrid': np.linspace(0, tf, N+1)}
        CB = DataCallback('CB', param)

        opts = {}
        opts['iteration_callback'] = CB
        opts['ipopt.max_iter'] = 10000

        prob = {'f': J, 'x': w, 'g': g}
        solver = nlpsol('solver', 'ipopt', prob, opts);

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        v_opt = sol['x']


    # Function to get x and u trajectories from w

    trajectories = Function('trajectories', [w], [x_plot, u_plot, x_thetas.reshape((3, -1))[:, :-1]], ['w'],
                                ['x', 'u', 'thetas'])

    x_opt, u_opt, thetas_opt = trajectories(v_opt)
    tgrid = np.linspace(0,tf,N)
    tgrid_radau = np.concatenate([tau_root + tk for tk in tgrid])

    if do_solve:
        x_plot, u_plot, thetas = CB.iter_sol_to_arrays(trajectories)
        x_plot.to_netcdf('../data/x_plot.nc')
        u_plot.to_netcdf('../data/u_plot.nc')
        thetas.to_netcdf('../data/thetas.nc')

    else:
        x_plot = xr.open_dataarray('../data/x_plot.nc')
        u_plot = xr.open_dataarray('../data/u_plot.nc')
        thetas = xr.open_dataarray('../data/thetas.nc')


    CP = collocation_plot(x_plot, u_plot, thetas, tgrid)

    fig, axs = CP.iteration_plot(5, full_plot=True)
    save = True
    if save:
        fig.savefig('../Figures/Collocation_Trajectory_' + sim_name + '.eps', format='eps')

    # CP.solution_plot(x_opt, u_opt)



