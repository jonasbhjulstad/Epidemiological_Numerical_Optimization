import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Parameters.Solver_Options import *
from casadi import *
from Callbacks.Singleshoot import Iteration_Callback
import pickle as pck
from Custom_Logging.Iteration_logging import setup_custom_logger
from Parameters.ODE_initial import N, M, tgrid, tgrid_M


def Single_Shooting_RK4(param, method='IPOPT'):
    if param == 'Social_Distancing':
        from Parameters.Parameters_Social_Distancing import x0, u_min, u_max, f, X_plot, Q_plot, u0
    elif param == 'Vaccination':
        from Parameters.Parameters_Vaccination_Flat import x0, u_min, u_max, f, X_plot, Q_plot, u0
    elif param == 'Isolation':
        from Parameters.Parameters_Isolation import x0, u_min, u_max, f, X_plot, Q_plot, u0
    else:
        return

    U = MX.sym('U', N)

    NV = U.shape[0]
    Ng = 0

    Q = f(x0, U)[1]

    prob = {'f': Q, 'x': U, 'g': []}

    CB = Iteration_Callback('Singleshoot_CB', NV, Ng,iter_step)

    if method == 'IPOPT':
        opts_IPOPT['iteration_callback'] = CB
        solver = nlpsol('solver', 'ipopt', prob, opts_IPOPT)
    elif method == 'SQP':
        opts_SQP['iteration_callback'] = CB
        solver = nlpsol('solver', 'sqpmethod', prob, opts_SQP)

    sol = solver(x0=[u0] * N, lbx=[u_min] * N, ubx=[u_max] * N, lbg=[], ubg=[])

    sim_data = {'U': CB.x_sols, 'lam_x': CB.lam_x_sols, 'lam_g': CB.lam_g_sols,
                'X': [X_plot(x0, u_sol) for u_sol in CB.x_sols],
                'Q': [Q_plot(x0, u_sol) for u_sol in CB.x_sols], 
                't': tgrid,
                't_M': tgrid_M, 'N': N, 'M': M, 'f': sol['f'], 'f_sols': CB.f_sols}

    fname = 'Single_Shooting_' + method + '_' + param
    with open(parent + '/data/' + fname + '.pck', 'wb') as file:
        pck.dump(sim_data, file)
    return fname

if __name__ == '__main__':
    Single_Shooting_RK4('Social_Distancing')
