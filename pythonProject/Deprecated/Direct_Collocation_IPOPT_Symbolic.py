# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from casadi import *
# from ODEs.SIR import SIR
import numpy as np
import matplotlib.pyplot as plt
from Collocation.collocation_coeffs import collocation_coeffs, collocation_polynomials
from Deprecated.IPOPT_Symbolical import ipopt_callback
from Callbacks.CasADI_Collocation_Callback import DataCallback
from Plot_tools.Collocation_Plot import collocation_plot
import pandas as pd
import ipopt
import pickle as pck
import xarray as xr
# Press the green button in the gutter to run the script.




if __name__ == '__main__':
    plt.close()


    d = 3
    B, C, D, tau_root = collocation_coeffs(d)

    do_solve = False
    data_path = r'../data/'


    from Parameters.Parameters_Vaccination_Flat import *

    # Start with an empty NLP
    w = []
    x_thetas = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = MX.sym('X0', 3)
    w.append(Xk)
    x_thetas.append(Xk)
    xi_min = [0]*3
    xi_max = [N_pop]*3
    x0 = [N_pop-I0, I0, 0]
    lbw.append(x0)
    ubw.append(x0)
    w0.append(x0)
    x_plot.append(Xk)

    u_min = 1e-5
    u_max = 1.1
    u0 = 1.0

    g_list = []
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w.append(Uk)
        lbw.append([u_min])
        ubw.append([u_max])
        w0.append([u_min])
        u_plot.append(Uk)
        gk = []

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = MX.sym('X_' + str(k) + '_' + str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            x_thetas.append(Xkj)
            lbw.append(xi_min)
            ubw.append(xi_max)
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
            gk.append(h * fj - xp)
            lbg.append([0]*nx)
            ubg.append([0]*nx)

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1];

            # Add contribution to quadrature function
            J = J + B[j] * qj * h

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k + 1), nx)
        w.append(Xk)
        x_thetas.append(Xk)
        lbw.append(xi_min)
        ubw.append(xi_max)
        w0.append(x0)
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end - Xk)
        gk.append(Xk_end-Xk)
        g_list.append(vertcat(*gk))
        lbg.append([0]*nx)
        ubg.append([0]*nx)

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
        compile = True
        N_workers = 2
        # Symbolic functions used in the cyipopt-callback:
        f_g = Function('g', [w], [g])
        grad_g = Function('grad_g', [w], [jacobian(g, w)])
        hess_g = [jacobian(jacobian(g[i], w), w) for i in range(Ng)]

        # Constraint-terms:
        lbd = MX.sym('lbd', Ng)
        hess_lbd_g = [lbd[i] * hess_gk for i, hess_gk in enumerate(hess_g)]
        hess_lbd_g_tot = 0
        for i in range(Ng):
            hess_lbd_g_tot += hess_lbd_g[i]

        # Cost-terms:
        obj = Function('obj', [w], [J])
        grad_obj = Function('grad_obj', [w], [jacobian(J, w)])
        hess_obj = jacobian(jacobian(J, w), w)

        # Total hessian:
        obj_factor = MX.sym('obj_factor')
        hess_total = Function('hess_total', [w, lbd, obj_factor], [obj_factor * hess_obj + hess_lbd_g_tot])


        def create_dict(*args):
            return dict({i: eval(i) for i in args})
        param = create_dict('nx','NW', 'Ng', 'f_g', 'grad_g', 'obj', 'grad_obj', 'hess_total', 'compile')


        CB = ipopt_callback(param)


        lbg = np.zeros(g.shape)
        ubg = np.zeros(g.shape)
        nlp = ipopt.problem(
            n=NW,
            m=Ng,
            problem_obj=CB,
            lb=lbw,
            ub=ubw,
            cl=lbg,
            cu=ubg
        )

        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-14)
        nlp.addOption('max_iter', 500)

        v_opt, info = nlp.solve(w0)
        with open(data_path + 'CB.pck', 'wb') as f:
            pck.dump(CB, f)
        with open(data_path + 'v_opt.pck', 'wb') as f:
            pck.dump(v_opt, f)



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

    CP.iteration_plot(10)
    # CP.solution_plot(x_opt, u_opt)



