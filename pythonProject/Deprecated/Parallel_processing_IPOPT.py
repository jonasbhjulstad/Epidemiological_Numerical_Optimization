import ipopt
import scipy.sparse as sps
import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
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
            self.compiled_functions_dict_list = []
            for i in range(self.N_workers):
                self.compiled_functions_dict_list.append(self.code_generation(i))

            pck.dump(self.compiled_functions_dict_list, open('fun_dict.obj', 'wb'))
            print("Code generation complete")
        else:
            self.compiled_functions_dict_list = pck.load(open('fun_dict.obj', 'rb'))

        self.worker_list = list(range(self.N_workers))

    def code_generation(self, i):
        g_list = self.g_split[i]
        J_list = self.J_split[i]
        filenames = ['grad_g_' + str(i) + '.c','hess_g_' + str(i) + '.c','grad_J_' + str(i) + '.c','hess_J_' + str(i) + '.c']
        executables = ['grad_g_' + str(i) + '.so','hess_g_' + str(i) + '.so','grad_J_' + str(i) + '.so','hess_J_' + str(i) + '.so']
        name_dict = {'filenames': executables}

        Generators = [CodeGenerator(fname) for fname in filenames]

        Wk = MX.sym('W', self.NW)
        f_namelist = []
        for k, (gk, Jk) in enumerate(zip(g_list, J_list)):
            k_str = str(k)
            f_names = ['grad_g_' + k_str, 'hess_g_' + str(k), 'grad_J_'+ k_str, 'hess_J_' + k_str]
            grad_g = Function(f_names[0], [Wk], [jacobian(gk(Wk), Wk)])
            hess_g = Function(f_names[1], [Wk], [jacobian(jacobian(gk(Wk), Wk), Wk)])
            grad_J = Function(f_names[2], [Wk], [jacobian(jacobian(Jk(Wk), Wk), Wk)])
            hess_J = Function(f_names[3], [Wk], [jacobian(jacobian(Jk(Wk), Wk), Wk)])
            Generators[0].add(grad_g)
            Generators[1].add(hess_g)
            Generators[2].add(grad_J)
            Generators[3].add(hess_J)
            f_namelist.append(f_names)

        [g.generate() for g in Generators]

        #Compile *.c to *.so shared libraries:
        [sp.run(['gcc', '-fPIC', '-shared', filename, '-o', executable]) for filename, executable in zip(filenames, executables)]

        name_dict['grad_g'] = [f_names[0] for f_names in f_namelist]
        name_dict['hess_g'] = [f_names[1] for f_names in f_namelist]
        name_dict['grad_J'] = [f_names[2] for f_names in f_namelist]
        name_dict['hess_J'] = [f_names[3] for f_names in f_namelist]

        name_dict['grad_g_executables'] = [external(fname, './' + executables[0]) for fname in name_dict['grad_g']]
        name_dict['hess_g_executables'] = [external(fname, './' + executables[1]) for fname in name_dict['hess_g']]
        name_dict['grad_J_executables'] = [external(fname, './' + executables[2]) for fname in name_dict['grad_J']]
        name_dict['hess_J_executables'] = [external(fname, './' + executables[3]) for fname in name_dict['hess_J']]

        return name_dict

    def objective(self, x):

        return self.obj(x).full()[0]


    def calculate_compiled_functions(self, x, worker_num, function_type):
        result_array = np.zeros((int(self.Ng/self.N_workers), self.NW))
        functions = self.compiled_functions_dict_list[worker_num][function_type]
        Ngk = functions[0](x).shape[0]
        results = np.zeros(Ngk)
        for i, f in enumerate(functions):
            results += f(x)
        return result_array
    def gradient(self, x):
        pool = mp.Pool(self.N_workers)

        results = pool.starmap(self.calculate_compiled_functions, zip([x]*self.N_workers, self.worker_list, ['grad_g_executables']*self.N_workers))

        return np.concatenate(results, axis=0)

    def constraints(self, x):

        return np.concatenate([g(x).full() for g in self.g], axis=0)

    def jacobian(self, x):
        pool = mp.Pool(self.N_workers)

        results = pool.starmap(self.calculate_compiled_functions, zip([x]*self.N_workers, self.worker_list, ['grad_g_executables']*self.N_workers))

        return np.concatenate(results, axis=0)

    def hessian(self, x, lagrange, obj_factor):
        pool = mp.Pool(self.N_workers)

        results_g = pool.starmap(self.calculate_compiled_functions, zip([x]*self.N_workers, self.worker_list, ['hess_g_executables']*self.N_workers))

        results_J = pool.starmap(self.calculate_compiled_functions, zip([x]*self.N_workers, self.worker_list, ['hess_J_executables']*self.N_workers))

        H = obj_factor*np.concatenate(results_g, axis=0)

        H+= np.concatenate(results_J, axis=0)


        return H



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