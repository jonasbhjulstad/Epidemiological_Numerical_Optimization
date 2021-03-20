
from casadi import *
from Callbacks.Singleshoot import Singleshoot_CB
from matplotlib import cm
from Parameters.Parameters_Social_Distancing import *
def MultiShootSQP_main(param, traj_initial=False, save=True):
    print("Solving ")
    if param == 'Social Distancing':
        from Parameters.Parameters_Social_Distancing import X, U, u_min, u_max, Wu, beta, xdot, x0, L, M, DT, h, f, X0, X_plot, u_lb, u_ub, u_init, u0, sim_name, Q, N, T, N_pop
    elif param == 'Isolation':
        from Parameters.Parameters_Social_Distancing import X, U, u_min, u_max, Wu, beta, xdot, x0, L, M, DT, h, f, X0, X_plot, u_lb, u_ub, u_init, u0, sim_name, Q, N, T, N_pop
    elif param == 'Vaccination':
        from Parameters.Parameters_Social_Distancing import X, U, u_min, u_max, Wu, beta, xdot, x0, L, M, DT, h, f, X0, X_plot, u_lb, u_ub, u_init, u0, sim_name, Q, N, T, N_pop


    nx = 3
    X0 = MX.sym('X0', nx)
    U = MX.sym('U')
    X = X0
    Q = 0
    nu = 1

    for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X+= DT/6*(k1 +2*k2 +2*k3 +k4)
       Q += DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = Function('F', [X0, U], [X, Q])




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

    # "Lift" initial conditions
    Xk = MX.sym('X0', 3)
    w += [Xk]
    lbw += x0
    ubw += x0
    w0 += x0
    x0_k = x0
    x_min = [0,0,0]
    x_max = [N_pop, N_pop, N_pop]
    Q = []
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k))
        w   += [Uk]
        lbw += [u_min]
        ubw += [u_max]
        w0  += [u0]

        # Integrate till the end of the interval
        Xk_end, Qk = F(Xk, Uk)
        x0_k, _ = F(x0_k, u0)
        Q.append(Qk)
        J=J+Qk

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), 3)
        w   += [Xk]
        lbw += x_min
        ubw += x_max
        if traj_initial:
            w0  += list(x0_k.full())
        else:
            w0 += x0
        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0,0,0]
        ubg += [0,0,0]

    Q_plot = Function('Q_plot', [vertcat(*w)], [vertcat(*Q)])
    opts = {}
    opts["expand"] = True
    opts['qpsol_options'] = {}
    opts['hessian_approximation'] = 'limited-memory'
    opts['qpsol_options']['enableFlippingBounds'] = True
    opts['max_iter'] = 1000
    opts['min_step_size'] = 1e-9
    opts['error_on_fail'] = False
    opts['qpsol_options']['error_on_fail'] = False

    iter_step = 1
    iter_file = '../data/log.opt'
    opts["calc_f"] = True
    opts["calc_g"] = True

    NV = N*(nx+nu) + nx
    Ng = len(g)*nx

    CB = Singleshoot_CB('Singleshoot_CB', NV, Ng,1,iter_step)
    opts["iteration_callback"] = CB
    opts["iteration_callback_step"] = iter_step


    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'sqpmethod', prob, opts)
    solvername = 'SQP'
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    fval = sol['f']
    w_opt = sol['x'].full().flatten()

    w_sols = CB.x_sols

    x0_sols = [sol[0::4] for sol in w_sols]
    x1_sols = [sol[1::4] for sol in w_sols]
    x2_sols = [sol[2::4] for sol in w_sols]
    u_sols = [sol[3::4] for sol in w_sols]

    lam_x_sols = CB.lam_x_sols

    lam_x0_sols = [sol[0::4] for sol in lam_x_sols]
    lam_x1_sols = [sol[1::4] for sol in lam_x_sols]
    lam_x2_sols = [sol[2::4] for sol in lam_x_sols]
    lam_u_sols = [sol[3::4] for sol in lam_x_sols]

    lam_g_sols = CB.lam_g_sols
    lam_g0_sols = [sol[0::3] for sol in lam_g_sols]
    lam_g1_sols = [sol[1::3] for sol in lam_g_sols]
    lam_g2_sols = [sol[2::3] for sol in lam_g_sols]


    lam_g_sols = CB.lam_g_sols


    N_sols = len(u_sols)

    tgrid = [T/N*k for k in range(N+1)]
    import matplotlib.pyplot as plt
    from matplotlib import cm
    colormap = cm.get_cmap('Greys', len(w_sols))
    colors = colormap(np.linspace(.1, .8, len(w_sols)))

    fig, axs = plt.subplots(4,1)

    for i in range(0, len(w_sols), 1):
        axs[0].plot(tgrid, x0_sols[i], '-', color = colors[i])
        axs[1].plot(tgrid, x1_sols[i], '-', color = colors[i])
        axs[2].plot(tgrid, x2_sols[i], '-', color = colors[i])
        axs[3].step(tgrid, vertcat(DM.nan(1), u_sols[i]), '-', color = colors[i])


    axs[0].plot(tgrid, x0_sols[-1], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[1].plot(tgrid, x1_sols[-1], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[2].plot(tgrid, x2_sols[-1], linestyle='',marker='o',markersize=2.5, color = 'k')
    axs[3].step(tgrid, vertcat(DM.nan(1), u_sols[i]), linestyle='',marker='o',markersize=2.5, color='k')

    axs[0].set_ylabel('S')
    axs[1].set_ylabel('I')
    axs[2].set_ylabel('R')
    axs[3].set_ylabel('u')

    axs[0].set_title('RK4 Multiple-Shooting N = %i, ' %N+ "M = %i" %M+ ", f = {:.2e}".format(fval.full()[0][0]) + ", iterations = %i" %len(w_sols))

    axs[3].set_xlabel('time[days]')
    _ = [x.set_xticklabels([]) for x in axs[:-1]]
    _ = [x.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) for x in axs[:-1]]


    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()




    fig2, ax2 = plt.subplots(4)
    colors = colormap(np.linspace(.4, .8, len(w_sols)))

    for i in range(0, len(w_sols), 5):
        ax2[0].plot(tgrid, lam_x0_sols[i], '-', color = colors[i])
        ax2[1].plot(tgrid, lam_x1_sols[i], '-', color = colors[i])
        ax2[2].plot(tgrid, lam_x2_sols[i], '-', color = colors[i])
        ax2[3].plot(tgrid, vertcat(DM.nan(1), lam_u_sols[i]), '-', color = colors[i])

    ax2[0].plot(tgrid, lam_x0_sols[i], '-', color = 'k', marker='o', markersize=2.5)
    ax2[1].plot(tgrid, lam_x1_sols[i], '-', color = 'k', marker='o', markersize=2.5)
    ax2[2].plot(tgrid, lam_x2_sols[i], '-', color = 'k', marker='o', markersize=2.5)
    ax2[3].plot(tgrid, vertcat(DM.nan(1),lam_u_sols[i]), '-', color = 'k', marker='o', markersize=2.5)

    ax2[0].set_title(r'Multipliers for bounds (' + solvername + ', ' + sim_name + ')')
    _ = [x.set_ylabel(s) for s, x in zip(['S', 'I', 'R', 'u'], ax2)]
    # _ = [x.set_yscale('log') for x in ax2]
    fig2.subplots_adjust(hspace=.2)
    _ = [x.set_xticklabels('') for x in ax2[:-1]]
    _ = [x.set_xlabel('') for x in ax2[:-1]]
    ax2[-1].set_xlabel('time[days]')
    _ = [x.grid() for x in ax2]

    colors = colormap(np.linspace(.25, .8, len(w_sols)))

    fig3, ax3 = plt.subplots(4)
    for i in range(0, len(w_sols), 5):

        ax3[0].plot(tgrid, vertcat(DM.nan(1), lam_g0_sols[i]), '-', color = colors[i])
        ax3[1].plot(tgrid, vertcat(DM.nan(1), lam_g1_sols[i]), '-', color = colors[i])
        ax3[2].plot(tgrid, vertcat(DM.nan(1), lam_g2_sols[i]), '-', color = colors[i])
        ax3[3].plot(tgrid, vertcat(DM.nan(1),Q_plot(w_sols[i])), '-', color = colors[i])

    ax3[0].plot(tgrid, vertcat(DM.nan(1), lam_g0_sols[i]), '-', color = 'k', marker='o', markersize=2.5)
    ax3[1].plot(tgrid, vertcat(DM.nan(1), lam_g1_sols[i]), '-', color = 'k', marker='o', markersize=2.5)
    ax3[2].plot(tgrid, vertcat(DM.nan(1), lam_g2_sols[i]), '-', color = 'k', marker='o', markersize=2.5)
    ax3[3].plot(tgrid, vertcat(DM.nan(1),Q_plot(w_sols[i])), '-', color = 'k', marker='o', markersize=2.5)


    ax3[0].set_title(r'Multipliers for constraints')
    fig3.subplots_adjust(hspace=.5)
    ax3[-1].set_title(r'Objective values')
    _ = [x.set_xticklabels('') for x in ax3[:-1]]
    _ = [x.set_ylabel(s) for s, x in zip(['S', 'I', 'R'], ax3[:-1])]
    _ = [x.set_xlabel('') for x in ax3[:-1]]
    ax3[-1].set_xlabel('time[days]')
    _ = [x.grid() for x in ax3]

    plt.show()


    if traj_initial and save:

        fig.savefig('../Figures/Multiple_Shooting_Trajectory_SQP_traj_initial_' + sim_name + '.eps', format='eps')

        fig2.savefig('../Figures/Multiple_Shooting_bounds_SQP_traj_initial_' + sim_name + '.eps', format='eps')

        fig3.savefig('../Figures/Multiple_Shooting_obj_con_SQP_traj_initial_' + sim_name + '.eps', format='eps')

    elif save:
        fig.savefig('../Figures/Multiple_Shooting_Trajectory_SQP_' + sim_name + '.eps', format='eps')

        fig2.savefig('../Figures/Multiple_Shooting_bounds_SQP_' + sim_name + '.eps', format='eps')

        fig3.savefig('../Figures/Multiple_Shooting_obj_con_SQP_' + sim_name + '.eps', format='eps')