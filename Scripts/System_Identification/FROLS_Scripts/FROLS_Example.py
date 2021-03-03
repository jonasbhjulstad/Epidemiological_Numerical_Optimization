import numpy as np
from itertools import combinations, chain, tee
from FROLS_Scripts.FROLS import FROLS_volterra, bilinear_term, bilinear_term_generator
import matplotlib.pyplot as plt
N_pop = 5.3e6
I0 = 2000
x0 = np.array([[N_pop, N_pop - I0, 0]])
alpha = 1./9
R0 = 0.7

def SIR_discrete(x, R0):
    S = x[0]
    I = x[1]
    R = x[2]

    u = R0 * alpha
    return np.array([S - u*S*I/N_pop,
                     I + u*S*I/N_pop - alpha*I,
                     R + alpha*I])

def generate_data(magnitude=1, loc=0, sigma=1, span=[0,.33], Nu = 1000, param=[2, 3, 2, 2]):
    t = np.linspace(0, 28, Nu)
    R0_t = R0*(1 + np.sin(t*(2*np.pi)/t[-1]))

    X = np.concatenate([x0, np.zeros((Nu, 3))], axis=0)

    for i, r in enumerate(R0_t):
        X[i+1,:] = SIR_discrete(X[i, :], r)
    X_noise = X + magnitude * np.random.laplace(loc, sigma, X.shape)

    return X, X_noise, R0_t

if __name__ == '__main__':

    # np.seterr(all='raise')



    X, X_noise, u = generate_data()

    c = bilinear_term_generator([4,4], 3, X.shape[0])


    y = X[:-1, :]
    X_LS = np.concatenate([X[1:,:], u.reshape((-1,1))], axis=1)
    LS_params = np.linalg.inv(X_LS.T @ X_LS) @ (X_LS.T @ y)


    y = X[:,1]
    rho = 0.0000001
    g, q, ERR_list, ells, A, p, chosen_f_idxs = FROLS_volterra(X, y, rho)

    fig, ax = plt.subplots(2)
    ax[0].plot(y)
    ax[1].plot(g@q)




