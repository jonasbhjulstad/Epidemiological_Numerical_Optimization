import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Parameters.Solver_Options import *
from casadi import *
from Callbacks.Singleshoot import Iteration_Callback
import pickle as pck


def Multiple_Shooting_RK4(param, method='IPOPT', traj_initial=True):
    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import x0, u_min, u_max, u0, f, fk, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import x0, u_min, u_max, u0, f, fk, X_plot, Q_plot, N, M, nx, N_pop, tgrid_M, tgrid
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import x0, u_min, u_max, u0, f, fk, X_plot, Q_plot, N, nx, M, N_pop, tgrid_M, tgrid
    else:
        return

    U = MX.sym('U', N)
    X = MX.sym('X', (nx, N))

    # Constraint setup

    Q = 0
    g = [X[:, 0] - x0]
    Xk = x0
    for i in range(N - 1):
        Xk, Qk = fk(Xk, U[i])
        g.append(X[:, i + 1] - Xk)
        Q += Qk
    V = vertcat(U, X.reshape((-1, 1)))
    g = vertcat(*g)
    NV = V.shape[0]
    Ng = g.shape[0]

    Q = f(x0, U)[1]

    prob = {'f': Q, 'x': V, 'g': g}

    CB = Iteration_Callback('Singleshoot_CB', NV, Ng, iter_step)

    if method == 'IPOPT':
        opts_IPOPT['iteration_callback'] = CB
        solver = nlpsol('solver', 'ipopt', prob, opts_IPOPT)
    elif method == 'SQP':
        opts_SQP['iteration_callback'] = CB
        solver = nlpsol('solver', 'sqpmethod', prob, opts_SQP)

    trajinit=''
    if traj_initial:
        X0 = f(x0, u0)[0].full().squeeze()[:-nx]
        trajinit= '_initial'
    else:
        X0 = x0 * N
    U0 = [u0]*N
    V0 = [*U0, *X0]
    ubv = [u_max] * N + [N_pop] * (nx * N)
    lbv = [u_min] * N + [0] * (nx * N)

    sol = solver(x0=V0, lbx=lbv, ubx=ubv, lbg=[0] * Ng, ubg=[0] * Ng)
    U_sols = [U0, *[v_sol[:N] for v_sol in CB.x_sols]]
    sim_data = {'U': U_sols, 'lam_x': CB.lam_x_sols, 'lam_g': CB.lam_g_sols,
                'X': [X_plot(x0, u) for u in U_sols],
                'Q': [Q_plot(x0, u) for u in U_sols],
                'X_raw': [V_sol[N:] for V_sol in CB.x_sols],
                't_M': tgrid_M, 't': tgrid, 'N': N, 'M': M, 'f': sol['f'], 'f_sols': CB.f_sols}
    
    fname = 'Multiple_Shooting_' + method + '_' + param + trajinit
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)

    return fname



if __name__ == '__main__':
    Multiple_Shooting_RK4('Social_Distancing', method='IPOPT', traj_initial=True)
