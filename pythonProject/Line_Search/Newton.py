import numpy.linalg as la
from casadi import *
import logging
def armaijo(f, grad_f, w0, alpha=1, beta=.9):
    wk = w0
    delta_w = - la.inv(grad_f(wk)) @ f(wk)
    t = 1
    fk = f(wk)
    while norm_1(f(wk + alpha * t * delta_w)) >= norm_1(f(wk) + alpha * t * grad_f(wk).T @ delta_w):
        t *= beta
    wk = wk + alpha * t * delta_w
    return wk

def armaijo_newton(f, grad_f, w0, tol=1e-3, max_iter=100, logger=[]):
    wk = w0
    fk = f(wk)
    error = norm_1(fk)
    wk_list = []
    iter = 0
    while (error > tol) and (iter < max_iter):
        wk = armaijo(f, grad_f, wk)
        fk = f(wk)
        error = norm_1(fk)
        assert not np.isnan(error), 'NaN detected in newton iteration'
        if logger != []:
            logger.debug(error)
        wk_list.append(wk)
        iter+=1
    return wk_list

def newton_rhapson(f, grad_f, w0, tol=1e-3, max_iter=100, logger=[]):
    wk = w0
    fk = f(wk)
    error = norm_1(fk)
    wk_list = []
    iter = 0
    while (error > tol) and (iter < max_iter):
        fk = f(wk)
        error = norm_1(fk)
        wk = wk - la.inv(grad_f(wk)) @ fk
        assert not np.isnan(error), 'NaN detected in newton iteration'
        if logger != []:
            logger.debug(error)
        wk_list.append(wk)
        iter+=1
    return wk_list