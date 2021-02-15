import numpy as np
from ODEs.SIR import SIR
import matplotlib.pyplot as plt
import pickle as pck
import pandas as pd

def dF_dx(x, u, alpha):
    return np.array([[-u * x[1], -u * x[0], 0],
                     [u * x[1], u * x[0] - alpha, 0],
                     [0, alpha, 0]])


def dF_du(x):
    return np.array([-x[0] * x[1],
                     x[0] * x[1],
                     0])
def A_ODE(x, u, alpha, A):
    Adot = np.matmul(dF_dx(x, u, alpha), A)
    return Adot

def B_ODE(x, u, alpha, B):
    Bdot = np.matmul(dF_dx(x, u, alpha), B) + dF_du(x)
    return Bdot
def RK4(F, F_A, F_B, x, h, A, B):
    k1 = F(x)
    k2 = F(x + h/2*k1)
    k3 = F(x + h/2*k2)
    k4 = F(x + h*k3)
    xk_1 = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    k1_A = F_A(x, A)
    k2_A = F_A(x + h/2*k1, A)
    k3_A = F_A(x + h/2*k2, A)
    k4_A = F_A(x + h*k3, A)

    Ak_1 = A + h / 6 * (k1_A + 2 * k2_A + 2 * k3_A + k4_A)

    k1_B = F_B(x, B)
    k2_B = F_B(x + h/2*k1, B)
    k3_B = F_B(x + h/2*k2, B)
    k4_B = F_B(x + h*k3,B)

    Bk_1 = B + h / 6 * (k1_B + 2 * k2_B + 2 * k3_B + k4_B)

    return xk_1, Ak_1, Bk_1


def Variational_SIR_Sensitivity(dt,start=0 , stop=0,plot=False, save = False):

    X0 = np.array([997, 3, 0])
    N = np.sum(X0)
    alpha = 1.0/3.0
    R0 = 10
    u = R0*alpha/N

    F = lambda x: SIR(x, alpha, u)
    F_A = lambda x, S: A_ODE(x, u, alpha, S)
    F_B = lambda x, S: B_ODE(x, u, alpha, S)
    t = np.arange(0,18,dt)

    A = [np.eye(3)]
    x = [X0]
    B = [np.array([0,0,0])]

    l = 0
    for i, h in enumerate(np.diff(t)):
        if (i % (0.1 * len(t))) == 0:
            print('%d%%' % (l))
            l += 10
        xk_1, Ak_1, Bk_1 = RK4(F, F_A, F_B, x[-1], h, A[-1], B[-1])
        x.append(xk_1)
        A.append(Ak_1)
        B.append(Bk_1)

    S = [k[0] for k in x]
    I = [k[1] for k in x]
    R = [k[2] for k in x]

    #Sensitivities
    S_sense = [k.T[0] for k in A]
    I_sense = [k.T[1] for k in A]
    R_sense = [k.T[2] for k in A]

    if plot:
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(t, S, label="S")
        axs[0].plot(t, I, label="I")
        axs[0].plot(t, R, label = "R")
        axs[0].legend()

        axs[1].plot(t, S_sense)
        axs[1].legend(['S wrt. S', 'I wrt. I', 'I wrt. R'])
        axs[1].set_title('Sensitivities for S')

        axs[2].plot(t, I_sense)
        axs[2].legend(['I wrt. S', 'I wrt. I', 'I wrt. R'])
        axs[2].set_title('Sensitivities for I')

        u_sense = [[k[0], k[1]] for k in B]


        axs[3].plot(t, u_sense)
        axs[3].legend(['S wrt u', 'I wrt u'])

    plt.show()

    At = pd.Series(A, index=t)
    Bt = pd.Series(B, index=t)
    St = pd.Series(S, index=t)
    It = pd.Series(I, index=t)
    Rt = pd.Series(R, index=t)

    d = {'A': At, 'B': Bt, 'S': St, 'I': It, 'R': Rt}

    df = pd.DataFrame(d, index=t)
    if save:
        df.to_pickle('../data/Variational_sense.pck')

    return df