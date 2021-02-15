import cyipopt
import scipy.sparse as sps
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
import multiprocessing as mp
import subprocess as sp
import os
def list_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

class ipopt_callback(object):
    def __init__(self, param):
        #Assign parameter values to object:
        for key in param.keys():
            exec('self.'+key+' = '+'param[key]')


        self.g = param['g']
        self.J = param['J']

        self.g_split = list_split(self.g, self.N_workers)
        self.J_split = list_split(self.J, self.N_workers)

        if self.compile:
            print("Generating code for IPOPT-solver..")
            self.fnames = []
            for i in range(self.N_workers):
                self.fnames.append(self.code_generation(i))

            print("Code generation complete")
        self.pool = mp.Pool(self.N_workers)

    def code_generation(self, i):
        g_list = self.g_split[i]
        J_list = self.J_split[i]
        cname = 'worker_' + str(i) + '.c'
        names = {'filename': cname}
        C = CodeGenerator(cname)
        Wk = MX.sym('W', self.NW)
        for k, gk in enumerate(g_list):
            gname = 'grad_g_' + str(k)
            hname = 'hess_g_' + str(k)
            grad_g = Function(gname, [Wk], [jacobian(gk(Wk), Wk)])
            hess_g = Function(hname, [Wk], [jacobian(jacobian(gk(Wk), Wk), Wk)])
            C.add(grad_g)
            C.add(hess_g)
            gnames.append(gname)
            hnames.append(hname)

        names['gradients_g'] = gnames
        names['hessians_g'] = hnames

        gnames = []
        hnames = []
        for k, Jk in enumerate(J_list):
            gname = 'grad_J_' + str(k)
            hname = 'hess_J_' + str(k)
            grad_J = Function(gname, [Wk], [jacobian(jacobian(Jk(Wk), Wk), Wk)])
            hess_J = Function(hname, [Wk], [jacobian(jacobian(Jk(Wk), Wk), Wk)])
            C.add(grad_J)
            C.add(hess_J)


        names['gradients_J'] = gnames
        names['hessians_J'] = hnames

        C.generate()
        compiled_name = 'worker_'+str(i)+'.so'
        sp.run(['gcc', '-fPIC', '-shared', cname, '-o', compiled_name])
        # os.remove(cname)
        print("Worker_"+str(i) + " finished generating " + compiled_name)
        return names

    def objective(self, x):

        return self.obj(x).full()[0]

    def calculate_compiled_functions(self, x, filename, functions):
        grad_list = []
        for i, f in enumerate(functions):
            f = external(f, './' + filename)
            grad_list.append(f(x))
        return grad_list
    def gradient(self, x):

        return 0

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        grad_list = []
        for i, namelist in enumerate(self.fnames):
            grad_list.append(self.grad_gk(x, namelist))

        return sum(grad_list)

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return self.grad_g(x).full()

    # def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        #
        # col = self.tot_hess_sparsity.col
        # row = self.tot_hess_sparsity.row
        #
        # return col, row
    def hessian_Jk(self, x, fnames):
        filename = fnames['filename']
        hess_list = []
        for i, f in enumerate(fnames['hessians_J']):
            f = external(f, './' + filename)
            hess_list.append(f(x))
        return hess_list

    def hessian_gk(self, x, fnames):
        filename = fnames['filename']
        hess_list = []
        for i, f in enumerate(fnames['hessians_g']):
            f = external(f, './' + filename)
            hess_list.append(f(x))
        return hess_list
    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        hess_list = []
        for i, namelist in enumerate(self.fnames):
            hess_list.append(self.hessian_gk(x, namelist))

        H = obj_factor*self.hess_obj(x)

        H+= self.lbd_hess_g(lagrange, x)


        return H[self.tot_hess_sparsity.row, self.tot_hess_sparsity.col]


if __name__ == '__main__':
    x0 = [1.0, 5.0, 5.0, 1.0]

    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]
    CB = hs071()

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=CB,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    x, info = nlp.solve(x0)