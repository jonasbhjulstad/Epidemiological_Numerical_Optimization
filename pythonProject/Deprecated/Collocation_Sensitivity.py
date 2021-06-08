# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from casadi import *
from ODEs.SIR import SIR
import numpy as np
import matplotlib.pyplot as plt
from Callbacks.CasADI_Collocation_Callback import DataCallback
from Collocation.collocation_coeffs import collocation_coeffs





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

    #print(T)
    t = SX.sym("t")
    u = SX.sym("u")
    x = SX.sym("x", 3)


    alpha = 0.1
    Wu = 0.01
    xdot = SIR(x, alpha, u)
    qdot = x[1]*x[1] + Wu/(u*u)

    f = Function('f', [t, x, u], [xdot, qdot])

    u_lb = np.array([1e-8])
    u_ub = np.array([1.0])
    u_init = np.array([1e-2])

    N_pop = 10000
    I0 = 200
    x0 = [N_pop-I0, I0, 0]
    x_min = [0,0,0]
    x_max = [N_pop, N_pop, N_pop]
    xi_min = x0
    xi_max = x0
    xf_min = [0,0,0]
    xf_max = [N_pop, N_pop, N_pop]

    nx = 3
    nu = 1

    NX = nk*(d+1)*nx
    NU = nk*nu
    NXF = nx
    NV = NX+NU+NXF

    V = MX.sym("V", NV)

    vars_lb = np.zeros(NV)
    vars_ub = np.zeros(NV)
    vars_init = np.zeros(NV)
    offset = 0

    X = np.resize(np.array([], dtype=MX), (nk+1, d+1))
    U = np.resize(np.array([], dtype=MX), nk)
    #Peturbation parameter:
    P = MX.sym('P', 3)
    for k in range(nk):
        for j in range(d+1):
            X[k,j] = V[offset:offset+nx] + P

            vars_init[offset:offset+nx] = x0

            if k==0 and j==0:
                vars_lb[offset:offset+nx] = x0
                vars_ub[offset:offset+nx] = x0
            else:
                vars_lb[offset:offset + nx] = x_min
                vars_ub[offset:offset + nx] = x_max
            offset +=nx

        U[k] = V[offset:offset+nu]
        vars_lb[offset:offset+nu] = u_lb
        vars_ub[offset:offset+nu] = u_ub
        vars_init[offset:offset+nu] = u_init
        offset += nu
    X[nk, 0] = V[offset:offset+nx] + P
    vars_lb[offset:offset+nx] = xf_min
    vars_ub[offset:offset+nx] = xf_max
    vars_init[offset:offset+nx] = x0

    offset += nx

    g = []
    lbg = []
    ubg = []

    J = 0

    for k in range(nk):

        for j in range(1, d+1):
            xp_jk = 0
            for r in range(d+1):
                xp_jk += C[r,j]*X[k,r]

            fk, qk = f(T[k,j], X[k,j], U[k])
            g.append(h*fk - xp_jk)
            lbg.append(np.zeros(nx))
            ubg.append(np.zeros(nx))

            J += F[j]*qk*h


        xf_k = 0
        for r in range(d+1):
            xf_k += D[r]*X[k,r]

        g.append(X[k+1,0]-xf_k)
        lbg.append(np.zeros(nx))
        ubg.append(np.zeros(nx))

    g = vertcat(*g)
    Ng = g.shape[0]
    nlp = {'x':V, 'f':J, 'g':g, 'p': P}

    opts = {}
    opts["expand"] = True
    opts["ipopt"] = {}
    opts["ipopt"]["print_level"] = 3

    # qpsol_opts = dict(qpsol='qrqp', qpsol_options=dict(print_iter=False,error_on_fail=False), print_time=False)


    CB = DataCallback('DataCallback', NV, Ng, 0)
    opts["iteration_callback"] = CB

    solver = nlpsol("solver", "ipopt", nlp, opts)#qpmethod",nlp,qpsol_opts)
    # arg = {}

    # arg["x0"] = vars_init
    # arg["lbx"] = vars_lb
    # arg["ubx"] = vars_ub
    # arg["lbg"] = np.concatenate(lbg)
    # arg["ubg"] = np.concatenate(ubg)
    # arg["p"] = 0

    res = solver(x0=vars_init, lbx=vars_lb, ubx=vars_ub, lbg = np.concatenate(lbg), ubg=np.concatenate(ubg), p=0)

    # Retrieve the solution
    v_opt = np.array(res["x"])

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

    #High-level hessian calculation, low-level approach:
    nfwd = 3

    fwd_solver = solver.forward(nfwd)
    print('fwd_solver generated')

    fwd_lbx = [DM.zeros(res['x'].sparsity()) for i in range(nfwd)]
    fwd_ubx = [DM.zeros(res['x'].sparsity()) for i in range(nfwd)]
    fwd_p = [DM.zeros(P.sparsity()) for i in range(nfwd)]
    fwd_lbg = [DM.zeros(res['g'].sparsity()) for i in range(nfwd)]
    fwd_ubg = [DM.zeros(res['g'].sparsity()) for i in range(nfwd)]

    #Peturbation
    fwd_p[0][0] = 1
    fwd_p[1][1] = 1
    fwd_p[2][2] = 1


    # sol_fwd = fwd_solver(out_x=res['x'], out_lam_g=res['lam_g'], out_lam_x=res['lam_x'],
    #         out_f=res['f'], out_g=res['g'], lbx=vars_lb, ubx=vars_ub, lbg=lbg, ubg=ubg,
    #         fwd_lbx=horzcat(*fwd_lbx), fwd_ubx=horzcat(*fwd_ubx),
    #         fwd_lbg=horzcat(*fwd_lbg), fwd_ubg=horzcat(*fwd_ubg),
    #         p=0, fwd_p=horzcat(*fwd_p))

# Calculate the same thing using finite differences
h = 1e-3
pert = []

restest = solver(x0=res['x'], lam_g0=res['lam_g'], lam_x0=res['lam_x'],
                lbx=vars_lb + h*(fwd_lbx[0]+fwd_ubx[0]),
                ubx=vars_ub + h*(fwd_lbx[0]+fwd_ubx[0]),
                lbg=lbg + h*(fwd_lbg[0]+fwd_ubg[0]),
                ubg=ubg + h*(fwd_lbg[0]+fwd_ubg[0]),
                p=0 + h*fwd_p[0])
print(restest)
for d in range(nfwd):
    pert.append(solver(x0=res['x'], lam_g0=res['lam_g'], lam_x0=res['lam_x'],
                lbx=vars_lb + h*(fwd_lbx[d]+fwd_ubx[d]),
                ubx=vars_ub + h*(fwd_lbx[d]+fwd_ubx[d]),
                lbg=lbg + h*(fwd_lbg[d]+fwd_ubg[d]),
                ubg=ubg + h*(fwd_lbg[d]+fwd_ubg[d]),
                p=0 + h*fwd_p[d]))