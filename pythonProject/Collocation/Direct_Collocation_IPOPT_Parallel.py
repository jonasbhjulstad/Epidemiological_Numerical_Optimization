# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from casadi import *
# from ODEs.SIR import SIR
import numpy as np
import matplotlib.pyplot as plt
from Collocation.collocation_coeffs import collocation_coeffs
from Callbacks.Parallel_processing_IPOPT import  ipopt_callback
import pandas as pd
import ipopt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.close()
    d = 3

    C, D, F, tau_root = collocation_coeffs(d)
    nk = 100
    tf = 100

    h = tf/nk

    T = np.zeros((nk, d+1))
    for k in range(nk):
        for j in range(d+1):
            T[k,j] = h*(k+tau_root[j])

    u_lb = [1e-8]
    u_ub = [1.0]
    u_init = [1e-2]


    nx = 3
    nu = 1

    NX = nk*(d+1)*nx
    NU = nk*nu
    NXF = nx
    NV = NX+NU+NXF
    N_Wk = nx*(d+1) + nx

    alpha = 0.1
    Wu = 0.01
    N_pop = 10000

    #ODE-declaration for state and cost:

    def SIR(X, alpha, u):

        return vertcat(-u*X[0,:]*X[1,:], u*X[0,:]*X[1,:] - alpha*X[1,:], alpha*X[1,:])

    u = MX.sym("u")
    X = MX.sym("X",3, 3)
    X_next = MX.sym('X_next', 3,1)

    xdot = SIR(X, alpha, u)

    normalizing_factor = 1/(N_pop**2 + np.array(u_ub)**2)
    q = (X[1,:]*X[1,:] + Wu/(u*u))*normalizing_factor
    v = vertcat(X.reshape((-1,1)), u)

    qdot = jacobian(q, v)

    f = Function('f', [X, u], [xdot, qdot @ v])

    #Collocation constraints for a single trajectory:
    C = DM(C[1:,1:])
    D = DM(D[1:])
    F = DM(F[1:])
    Xk = 1/h * X @ C

    Fk, Qk = f(X, u)

    Jk = h*F.T@Qk

    Xf = X @ D

    g_col = Fk - Xk

    g_shoot = Xf-X_next
    gk = vertcat(g_col.reshape((-1,1)), g_shoot.reshape((-1,1)))

    Wk = vertcat(*[X.reshape((-1,1)), X_next, u])
    f_con = Function('f_con', [Wk], [gk])
    f_cost = Function('f_cost', [Wk], [Jk])


    #Full constraint declaration


    X = MX.sym('X', nx, (d+1)*(nk+1))
    U = MX.sym('U', 1,nk)

    NX = nx*(d+1)*(nk+1)
    NU = nk
    N_Wk = Wk.shape[0]
    NW = NX + NU
    g = []
    J = []
    W = MX.sym('W', NX + NU)
    for i in range(nk):
        Wk = W[i+nx:i+N_Wk+nx]
        g.append(Function('g' + str(i), [W], [f_con(Wk)]))
        J.append(Function('J' + str(i), [W], [f_cost(Wk)]))

    Ng = len(g)*f_con.numel_out(0)

    total_cost = sum([j(W) for j in J])
    obj = Function('J', [W], [total_cost])

    def create_dict(*args):
        return dict({i: eval(i) for i in args})

    compile = False
    N_workers = 2
    param = create_dict('g', 'J', 'nx', 'NX', 'NU','NW', 'N_Wk', 'Ng', 'N_workers', 'compile', 'obj')

    CB = ipopt_callback(param)

    I0 = 200
    x0 = [N_pop-I0, I0, 0]
    x_min = [0,0,0]
    x_max = [N_pop, N_pop, N_pop]
    xi_min = x0
    xi_max = x_max
    xf_min = [0,0,0]
    xf_max = [N_pop, N_pop, N_pop]



    W0_ub = x0 + x_max*3 + u_ub
    Wf_ub = x_max*3 + x0
    Wk_ub = x_max*4 + u_ub
    W_ub =  W0_ub + Wk_ub*(nk-1) + Wf_ub

    W0_lb = x0 + x_min*3 + u_lb
    Wk_lb = x_min*4 + u_lb
    Wf_lb = x_min*3 + x0
    W_lb = W0_lb + Wk_lb*(nk-1) + Wf_lb

    lbg = [0]*Ng
    ubg = [0]*Ng

    Wk_0 = x0*4 + u_init
    W0 = Wk_0*nk + x0*4

    nlp = ipopt.problem(
        n=NW,
        m=Ng,
        problem_obj=CB,
        lb=W_lb,
        ub=W_ub,
        cl=lbg,
        cu=ubg
    )

    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-14)

    x, info = nlp.solve(W0)




    # Retrieve the solution
    v_opt = x

    # Get values at the beginning of each finite element
    x0_opt = v_opt[0::(d + 1) * nx + nu]
    x1_opt = v_opt[1::(d + 1) * nx + nu]
    x2_opt = v_opt[2::(d + 1) * nx + nu]
    u_opt = v_opt[(d + 1) * nx::(d + 1) * nx + nu]
    tgrid = np.linspace(0, tf, nk + 1)
    tgrid_u = np.linspace(0, tf, nk)

    # Plot the results
    fig, ax = plt.subplots(2,1)
    ax[0].plot(tgrid, x0_opt, '--')
    ax[0].plot(tgrid, x1_opt, '-.')
    ax[0].plot(tgrid, x2_opt, '-.')
    ax[0].set_title("SIR Collocation Optimization")
    ax[1].ticklabel_format(useMathText=True)
    ax[1].step(tgrid_u, u_opt, '-')

    ax[0].set_xlabel('time')
    ax[0].legend(['S trajectory', 'I trajectory', 'R trajectory'])
    ax[1].legend(['u trajectory'])
    ax[0].grid()
    plt.show()

    #Extract Iteration data from CallBack:
    CB_data = {'f': CB.f_sols,'S': CB.S, 'I': CB.I, 'R':CB.R,'u': CB.u,'g': CB.g_sols, 'lam_x': CB.lam_x_sols,
               'lam_g': CB.lam_g_sols, 'iter': CB.iter, 'I_thetas': CB.I_thetas}
    df_cas = pd.DataFrame(CB_data).set_index('iter')
    # Convert Iteration data:
    # df_ipopt = parse_IPOPT_log(iter_file, r'C:/Users/jonas/OneDrive/Dokumenter/ntnu_Covid/Data/Direct_Collocation_Iteration_data.pck')

    # df = df_ipopt.merge(df_cas, left_on='iter', right_on='iter')
    df['Wu'] = Wu
    df['t'] = [tgrid]*df.shape[0]
    df['N'] = N_pop

    df.to_pickle('../data/collocation_data.pck')

