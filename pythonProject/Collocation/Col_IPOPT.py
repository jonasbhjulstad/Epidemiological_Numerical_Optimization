''
from casadi import *
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
import numpy as np
import matplotlib.pyplot as plt
from Collocation.collocation_coeffs import collocation_coeffs

import pickle as pck
# Press the green button in the gutter to run the script.
from Parameters.ODE_initial import T, h
from RK4.Integrator import RK4_Integrator
from Callbacks.Singleshoot import Iteration_Callback
from Parameters.ODE_initial import *
from Parameters.Solver_Options import *


def Direct_Collocation(param, method='IPOPT'):
    if param == 'Social_Distancing':
            from Parameters.Parameters_Social_Distancing import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, nx, M, N_pop, tgrid_M, tgrid
    else:
        return 
    # Continuous time dynamics

    # Degree of interpolating polynomial
    d = 1

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'radau'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)


    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    gu = []
    gth = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = MX.sym('X0', nx)
    w.append(Xk)
    lbw.append([0, 0,0])
    ubw.append([N_pop, N_pop, N_pop])
    w0.append(x0)
    g.append(Xk - x0)
    lbg.append([0,0,0])
    ubg.append([0,0,0])
    x_plot.append(Xk)
    xk = x0
    Q = []
    Xth = []
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w.append(Uk)
        lbw.append([u_min])
        ubw.append([u_max])
        w0.append([u0])
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-N_pop, -N_pop, -N_pop])
            ubw.append([N_pop,  N_pop, N_pop])
            w0.extend(F(x0, u0)[0].full())
        Xth.extend(Xc)
        # Loop over collocation points
        Xk_end = D[0]*Xk
        Qk = 0
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): 
                xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            fj, qj = F(Xc[j-1],Uk)

            gth.append(h*fj - xp)
            g.append(h*fj - xp)
            lbg.append([0, 0,0])
            ubg.append([0, 0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1];

            # Add contribution to quadrature function
            Qk += B[j]*qj*h
        Q.append(Qk)
        J += Qk
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
        lbw.append([0, 0, 0])
        ubw.append([N_pop, N_pop, N_pop])
        xk, _ = fk(xk, u0)
        w0.extend(xk.full())
        x_plot.append(Xk)

        # Add equality constraint
        gu.append(Xk_end-Xk)
        g.append(Xk_end-Xk)
        lbg.append([0, 0, 0])
        ubg.append([0, 0, 0])

    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)
    x_plot = horzcat(*x_plot)
    u_plot = vertcat(*u_plot)
    Xth = horzcat(*Xth).reshape((-1,1))
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    CB = Iteration_Callback('Singleshoot_CB', w.shape[0], g.shape[0], iter_step)

    prob = {'f': J, 'x': w, 'g': g}

    if method == 'IPOPT':
        opts_IPOPT['iteration_callback'] = CB
        solver = nlpsol('solver', 'ipopt', prob, opts_IPOPT)
    elif method == 'SQP':
        opts_SQP['iteration_callback'] = CB
        solver = nlpsol('solver', 'sqpmethod', prob, opts_SQP)
    # Create an NLP solver

    # Function to get x and u trajectories from w
    trajectories = Function('trajectories', [w], [x_plot, u_plot, Xth])


    obj_val = Function('objval', [w], [vertcat(*Q)])
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    X_sols = []
    U_sols = []
    Q_sols = []
    th_sols = []
    lxx = []
    lxu = []
    lxth = []
    lgx = []
    lgu = []
    lgth = []
    for v_sol, lam_x_sol, lam_g_sol in zip(CB.x_sols, CB.lam_x_sols, CB.lam_g_sols):
        x_sol, u_sol, th_sol = trajectories(v_sol)
        lam_x_x, lam_x_u, lam_x_th = trajectories(lam_x_sol)
        X_sols.append(x_sol.full())
        th_sols.append(th_sol)

        U_sols.append(u_sol.full())
        Q_sols.append(obj_val(v_sol))
        lxx.append(lam_x_x)
        lxu.append(lam_x_u)
        lxth.append(lam_x_th)

    sim_data = {'U': U_sols, 'lxx': lxx, 'lxu': lxu, 'lxth': lxth, 'lam_g': CB.lam_g_sols, 'lam_x': CB.lam_x_sols,
            'X': X_sols,'Q': Q_sols, 'd': d,
            't_M': tgrid_M, 't': tgrid, 
            'N': N, 'M': M, 'f': sol['f'], 
            'f_sols': CB.f_sols}

    fname = 'Direct_Collocation_' + method + '_' + param + '_initial'
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)

    
    return fname
if __name__ == '__main__':
    Direct_Collocation('Isolation')