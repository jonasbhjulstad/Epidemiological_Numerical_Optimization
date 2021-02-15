import matplotlib.pyplot as plt
from Collocation.collocation_coeffs import collocation_coeffs
from matplotlib.patches import Polygon

import numpy as np
from casadi import *
def Collocation_Trajectory(d, theta,tau_root):
    s = np.poly1d([0.0,0.0,0.0,0.0])
    for j in range(d+1):
        p = np.poly1d([1])
        for r in range(d+1):
            if r!=j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
        s = s + theta[j]*p
    return s
if __name__ == '__main__':
    d = 3
    C, D, F, tau_root = collocation_coeffs(d)
    t = np.linspace(0,1,100)

    t_der = np.linspace(-.1,.1,20)

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)
    p_list = []
    pder_list = []
    pint_list = []
    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        p_list.append(p)

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])
        pder_list.append(pder)
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        pint_list.append(pint)
        B[j] = pint(1.0)

    pol_traj = [pol(t) for pol in p_list]
    y_max = max([max(pol) for pol in pol_traj])
    y_min = min([min(pol) for pol in pol_traj])
    t_y = np.linspace(y_min, y_max, 100)

    fig, ax = plt.subplots(4)
    [ax[i].plot(t, pol, color='k') for i, pol in enumerate(pol_traj)]
    tau_plot = np.repeat(np.array(tau_root).reshape((-1,1)), 100, axis=1).T
    [ax[i].plot(tau_plot, t_y, color='gray', label='tau') for i in range(len(ax))]
    [x.grid() for x in ax]
    theta = [[p(t) for t in tau_root] for p in p_list]
    d_list = []
    for j in range(d+1):
        jd_list = []
        for r in range(d+1):
            d_line = p_list[j](tau_root[r]) + pder_list[j](tau_root[r])*t_der
            jd_list.append(d_line)
        d_list.append(jd_list)
    [[ax[j].plot(t_der + tau_root[r], d_list[j][r], '--', color='k') for r in range(d+1)] for j in range(d+1)]
    Xf = np.array(theta) @ D
    [ax[j].plot(1, Xf[j], 'o', 'k', label='$theta_{d+1}$') for j in range(len(ax))]

    # Make the shaded region
    ix = np.linspace(0, 1)
    iy_list = [p(ix) for p in p_list]

    a = 0
    b = 1
    vert_list = [[(a, 0), *zip(ix, iy), (b, 0)] for iy in iy_list]
    polys = [Polygon(verts, facecolor='0.9', edgecolor='0.5') for verts in vert_list]
    [x.add_patch(poly) for x, poly in zip(ax, polys)]
    # [ax[i].set_title('Polynomial %i' %i) for i in range(d+1)]
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    custom_lines = [Line2D([0], [0], color='gray'),
                    Line2D([0], [0], linestyle='--', color='k'),
                    Line2D([0], [0], color='k'),
                    Line2D([0], [0], linestyle='',marker='o', color='k'),
                    Patch(facecolor='lightgray',
                          label='Color Patch')
                    ]
    [ax[i].legend(custom_lines, [r'$\tau$',r'$C_j \theta$', r'$P_j(t)$', r'$D_j\theta$', r'$B_j$ integral area'], loc='lower left') for i in [3]]

    [ax[i].set_ylabel(r'$P_%i(t)$' %i) for i in range(d+1)]
    ax[-1].set_xlabel('t')
    plt.show()

    # plt.savefig(r'../Plot/lagrange_polynomial.eps', format='eps')
