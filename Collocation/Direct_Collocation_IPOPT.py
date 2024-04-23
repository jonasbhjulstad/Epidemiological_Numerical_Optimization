
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

def Initial_Trajectory(F, N, d, x0, U, h, tau_root):
    tau_size = np.diff(tau_root)
    x0 = np.reshape(x0, (-1,1))
    Xkj = x0
    Xdot_list = []
    X0 = [x0]
    for k in range(N):
        Xdot_list = []
        for j in range(d):
            Xkj = RK4_Integrator(F, Xkj, U[k], h*tau_size[j])
            if j != d-1:
                Xdot_list.append(F(Xkj, U[k])[0].full())
            else:
                
                X0.append(Xkj)
                Xdot_list.append(F(Xkj, U[k])[0].full())
            

        X0.append(np.concatenate(Xdot_list, axis=1))
    return np.concatenate(X0, axis=1).reshape((-1))


def Direct_Collocation(param, method='IPOPT'):


    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import x0, u_min, u_max, u0, fk, F, X_plot, Q_plot, N, nx, M, N_pop, tgrid_M, tgrid
    else:
        return

    d = 3
    B, C, D, tau_root = collocation_coeffs(d)

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'legendre'))

    J = 0

    u0 = u_min

    U = MX.sym('U', N)

    #Control timepoints for states + endstate:
    Xu = [MX.sym('Xu_' + str(i), (nx,1)) for i in range(N+1)]
    #Thetas excluding control timepoint states:
    Xth = [MX.sym('Xth_' + str(i), (nx,d)) for i in range(N)] 

    #Compute xdot, Qk for all thetas:
    Fth = F.map(d+1, 'serial')
    gth = []
    gu = [x0-Xu[0]]
    # Formulate the NLP
    for k in range(N):
        #All states in current iteration
        Xk = horzcat(*[Xu[k], Xth[k]])
        # Loop over collocation points
        Xk_end = D[0] * Xu[k]

        # Expression for the state derivative at the collocation point
        Xp = C.T @ Xk.T
        Xk_end = D.T @ Xk.T

        # Append collocation equations
        Xth_next = vertcat(*[F(Xk[:,i], U[k])[0] for i in range(Xk.shape[1])])
        # Xth_next, _ = Fth(Xk, horzcat(*([U[k]]*N)))
        gth.append((h * Xth_next[nx:] - Xp[nx:]))
        # Add contribution to quadrature function        J = J + B.T @ Qk.T * h

        # Add equality constraint
        gu.append((Xk_end.T - Xu[k]).reshape((-1,1)))


    g = vertcat(*[vertcat(*gth), vertcat(*gu)])
    X = [Xu[0]]
    X.extend([horzcat(u, th) for u, th in zip(Xu[1:], Xth)])   

    X = horzcat(*X)

    NX = X[:].shape[0]
    V = vertcat(*[U, X.reshape((-1,1))])

    Ng = g.shape[0]


    #Construct Initial Trajectory
    traj_initial=True
    if traj_initial:
        X0 = Initial_Trajectory(F, N, d, x0, [u0]*N, h, tau_root)
        # X0 = f(x0, u0)[0].full().squeeze()[:-nx]
        trajinit= '_initial'
    else:
        X0 = [x0]*(N*(d+1) + 1)

    V0 = np.concatenate([[u0]*N, X0], axis=0)

    lbg = ubg = [0]*Ng
    lbv = [u_min]*N + [-N_pop]*NX
    ubv = [u_max]*N + [N_pop]*NX

    CB = Iteration_Callback('Singleshoot_CB', V.shape[0], Ng, 1)
    opts = {}
    opts['iteration_callback'] = CB
    opts['ipopt.max_iter'] = 10000

    prob = {'f': J, 'x': V, 'g': g}
    solver = nlpsol('solver', 'ipopt', prob, opts);


    # Solve the NLP
    sol = solver(x0=V0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)


    separate_v = Function('sep_v', [V], [horzcat(*Xu), U, X])
    # separate_g = Function('sep_g', [g], [vertcat(*gth), vertcat(*gu)])

    X_sols = []
    U_sols = []
    Theta_sols = []
    lam_g_sols = []
    lam_x_sols = []
    for V_sol, lam_x_sol, lam_g_sol in zip(CB.x_sols, CB.lam_x_sols, CB.lam_g_sols):
        X_sol, U_sol, Theta_sol = separate_v(V_sol)
        X_sols.append(X_sol.full().T)
        U_sols.append(U_sol.full())
        Theta_sols.append(Theta_sol.full())
        lam_x_sols.append(separate_v(lam_x_sol))
        # lam_g_sols.append(separate_g(lam_g_sol)) 
    

    a = CB.x_sols[0]
    sim_data = {'U': U_sols, 'lam_x': lam_g_sols, 'lam_g': lam_x_sols,
                'X': X_sols,
                't': tgrid, 'N': N, 'f': CB.f_sols[-1]}
    fname = 'Collocation_' + method + '_' + param + trajinit
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)
    
    return fname
    


if __name__ == '__main__':
    Direct_Collocation('Isolation')
