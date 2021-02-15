import cyipopt
import scipy.sparse as sps
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
import multiprocessing as mp
import subprocess as sp

class ipopt_callback(object):
    def __init__(self, param):
        for key in param.keys():
            exec('self.'+key+' = '+'param[key]')

        print("Generating code for IPOPT-solver..")
        C = CodeGenerator('CB.c')
        C.add(self.obj)
        C.add(self.grad_obj)
        C.add(self.hess_total)
        C.add(self.f_g)
        C.add(self.grad_g)

        C.generate()
        print("Code generation complete")
        if self.compile:
            print("Compiling code to shared libraries..")
            sp.run(['gcc', '-fPIC', '-shared', './CB.c', '-o', './CB.so'])
            print("Compile complete, importing functions to class..")
        else:
            print("Warning: Compile is set to False, importing functions..")
        self.obj = external('obj', './CB.so')
        self.grad_obj = external('grad_obj', './CB.so')
        self.tot_hess = external('hess_total', './CB.so')
        self.g = external('g', './CB.so')
        self.grad_g = external('grad_g', './CB.so')

        self.tot_hess_sparsity = sps.coo_matrix(DM.ones(self.tot_hess.sparsity_out(0)).full())

        print("Functions imported")


    def objective(self, x):

        return self.obj(x).full()[0]

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return self.grad_obj(x).full()

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return self.g(x).full()

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return self.grad_g(x).full()

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #

        col = self.tot_hess_sparsity.col
        row = self.tot_hess_sparsity.row

        return col, row

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = self.hess_total(x, lagrange, obj_factor)


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