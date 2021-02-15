from scipy.optimize import newton
import numpy as np
from numpy import matmul as mul
from scipy.optimize import newton
from scipy.linalg import cho_factor, cho_solve
import numpy.linalg as la

if __name__ == '__main__':
    nx = 3
    nu = 1

    NX = 10
    NU = 5

    N_pop = 10000
    I0 = 200
    x0 = [N_pop-I0, I0, 0]

    w0 = np.array([[1],[2]])
    u0 = 0.1
    S = 1
    Q = np.eye(2)

    def Phi(w):
        K = mul(1/2*(w-w0).T,Q)
        return mul(K,w-w0)

    def Nabla_Phi(w):
        return mul(Q ,w-w0)

    def Hess_Phi(w):
        return Q

    def h(w):
        K = w.T*S
        return mul(K,(w - 1))
    def Nabla_h(w):
        return 2*S*w

    def r(x,tau):
        Nw = 2
        N_mu = 1
        N_s = 1
        w = x[:Nw]
        mu = x[Nw:Nw+N_mu]
        s = x[Nw+N_mu:Nw+N_mu+N_s]
        return np.concatenate([Nabla_Phi(w), h(w) + s, mu*s-tau], axis=0)

    def nabla_r(x):
        Nw = 2
        N_mu = 1
        N_s = 1
        w = x[:Nw]
        mu = x[Nw:Nw+N_mu]
        s = x[Nw+N_mu:Nw+N_mu+N_s]

        H = Hess_Phi(w)
        grad_h = Nabla_h(w)
        S = np.diag(s).reshape(s.shape)
        Mu = np.diag(mu).reshape(mu.shape)
        grad_r = np.concatenate([np.concatenate([H, grad_h, np.zeros((grad_h.shape[0], N_mu))], axis=1),
                  np.concatenate([grad_h.T, np.zeros((1,1)), np.eye(N_mu)], axis=1),
                  np.concatenate([np.zeros((N_s, grad_h.shape[0])), S, Mu], axis=1)], axis=0)
        # grad_r = np.zeros((N_w + 2*N_mu, N_w + 2*N_mu))
        #
        # grad_r[0:N_w, 0:N_w] = Hess_Phi(w)
        #
        # grad_h = Nabla_h(w)
        # print(grad_h)
        # grad_r[0:N_w, N_w:N_w+N_mu] = grad_h
        # grad_r[N_w:N_w+N_mu, 0:N_w] = grad_h.T
        # grad_r[N_w+N_mu:N_w+N_mu + N_s, N_w+N_mu:N_w+N_mu] = np.diag(s)
        # grad_r[N_w+N_mu:N_w+N_mu + N_s, -N_mu:] = np.eye(N_mu)
        # grad_r[-N_mu:, -N_mu:] = np.diag(mu)

        return grad_r


    tau = 1
    mu = 1
    s = 1
    tol = 1e-4
    xk = np.array([[1],[3],[0.5],[0.5]])
    ntol = 1e-1
    iter = 0
    a = ntol+1
    while tau > tol:
        print("tau = {:.2f}".format(tau))
        while a > ntol:
            # print(iter)
            grad_r = nabla_r(xk)
            # c, low = cho_factor(grad_r)
            xk += -0.3*mul(la.inv(grad_r), r(xk,tau))
            a = la.norm(xk)
            print(xk)

        tau-=tau/3