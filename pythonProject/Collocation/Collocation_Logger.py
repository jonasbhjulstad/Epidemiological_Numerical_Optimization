from casadi import *
import numpy as np
class Collocation_Logger(object):
    def __init__(self, params):
        self.d = params['d']

        self.C, self.D, self.F, self.tau_root = collocation_coeffs(d)
        self.nk = params['nk']
        self.tf = params['tf']
        self.h = tf/nk

        self.T = np.zeros((self.nk, self.d + 1))
        for k in range(self.nk):
            for j in range(self.d + 1):
                T[k, j] = self.h * (self.k + self.tau_root[j])

        self.xdot = params['xdot']
        self.qdot = params['qdot']
        self.boundaries = {'u_lb': params['u_lb'],
                           'u_ub': params['u_ub'],
                           'xi_min': params['xi_min'],
                           'xi_max': params['xi_max'],
                           'x_min': params['x_min'],
                           'x_max': params['x_max'],
                           'xf_min': params['xf_min'],
                           'xf_max': params['xf_max']}
        self.init_vals = {'x0': params['x0'],
                          'u0': params['u0']}
        self.nx = params['nx']
        self.nu = params['nu']
        self.X = []
        self.U = []
        self.V = []

        self.NX = nk * (d + 1) * nx
        self.NU = nk * nu
        self.NXF = nx
        self.NV = NX + NU + NXF

        self.vars_lb = []
        self.vars_ub = []
        self.vars_init = []

        self.g = []
        self.lbg = []
        self.ubg = []

        self.J = 0

        self.Ng = 0

        self.nlp = []
        self.result = []

    def construct_nlp(self,opts,sol_name='ipopt', offset=0):
        self.X = np.resize(np.array([], dtype=MX), (self.nk + 1, self.d + 1))
        self.U = np.resize(np.array([], dtype=MX), self.nk)
        self.V = MX.sym("V", NV)

        self.vars_lb = np.zeros(NV)
        self.vars_ub = np.zeros(NV)
        self.vars_init = np.zeros(NV)
        for k in range(nk):
            for j in range(d + 1):
                self.X[k, j] = self.V[offset:offset + nx]

                self.vars_init[offset:offset + self.nx] = self.init_vals['x0']

                if k == 0 and j == 0:
                    self.vars_lb[offset:offset + nx] = self.init_vals['x0']
                    self.vars_ub[offset:offset + nx] = self.init_vals['x0']
                else:
                    self.vars_lb[offset:offset + nx] = self.boundaries['x_min']
                    self.vars_ub[offset:offset + nx] = self.boundaries['x_max']
                offset += self.nx

            self.U[k] = self.V[offset:offset + self.nu]
            self.vars_lb[offset:offset + self.nu] = self.boundaries['u_min']
            self.vars_ub[offset:offset + self.nu] = self.boundaries['u_max']
            self.vars_init[offset:offset + self.nu] = self.init_vals['u0']
            offset += nu
        self.X[self.nk, 0] = self.V[offset:offset + self.nx]
        self.vars_lb[offset:offset + self.nx] = self.boundaries['xf_min']
        self.vars_ub[offset:offset + self.nx] = self.boundaries['xf_max']
        self.vars_init[offset:offset + self.nx] = self.init_vals['x0']
        offset += self.nx

        for k in range(self.nk):

            for j in range(1, self.d + 1):
                xp_jk = 0
                for r in range(d + 1):
                    xp_jk += C[r, j] * X[k, r]

                fk, qk = f(T[k, j], self.X[k, j], self.U[k])
                self.g.append(self.h * fk - xp_jk)
                self.lbg.append(np.zeros(self.nx))
                self.ubg.append(np.zeros(self.nx))

                self.J += self.F[j] * qk * self.h

            xf_k = 0
            for r in range(d + 1):
                xf_k += self.D[r] * self.X[k, r]

            self.g.append(X[k + 1, 0] - xf_k)
            self.lbg.append(np.zeros(self.nx))
            self.ubg.append(np.zeros(self.nx))

        self.g = vertcat(*g)
        self.Ng = self.g.shape[0]
        nlp = {'x': V, 'f': J, 'g': g}
        solver = nlpsol("solver", sol_name, nlp, opts)
        return solver

        def solve():
            arg["x0"] = self.vars_init
            arg["lbx"] = self.vars_lb
            arg["ubx"] = self.vars_ub
            arg["lbg"] = np.concatenate(self.lbg)
            arg["ubg"] = np.concatenate(self.ubg)

        self.result = solver(**arg)
        return self.result

