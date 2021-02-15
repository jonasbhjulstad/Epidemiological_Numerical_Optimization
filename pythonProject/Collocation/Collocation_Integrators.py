# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from casadi import *
from ODEs.SIR import SIR
import numpy as np
import matplotlib.pyplot as plt
from Callbacks.CasADI_Collocation_Callback import DataCallback

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.close()
    d = 3

    nk = 100
    tf = 100

    h = tf/nk
    tau = collocation_points(d, "radau")

    for j in range(d+1):
        p = np.poly1d([1])

        for r in range(d+1):
            if r!= j:
                p *= np.poly1d([1, -tau_root[r]], [tau_root[j] - tau_root[r]])
        D[j] = p(1.0)

        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        pint = np.polyint(p)

        F[j] = pint(1.0)


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

    for k in range(nk):
        for j in range(d+1):
            X[k,j] = V[offset:offset+nx]

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
    X[nk, 0] = V[offset:offset+nx]
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
    nlp = {'x':V, 'f':J, 'g':g}

    opts = {}
    opts["expand"] = True
    opts["ipopt"] = {}
    opts["ipopt"]["print_level"] = 3

    CB = DataCallback('DataCallback', NV, Ng, 0)
    opts["iteration_callback"] = CB

    solver = nlpsol("solver", "ipopt", nlp, opts)
    arg = {}

    arg["x0"] = vars_init
    arg["lbx"] = vars_lb
    arg["ubx"] = vars_ub
    arg["lbg"] = np.concatenate(lbg)
    arg["ubg"] = np.concatenate(ubg)

    res = solver(**arg)

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


