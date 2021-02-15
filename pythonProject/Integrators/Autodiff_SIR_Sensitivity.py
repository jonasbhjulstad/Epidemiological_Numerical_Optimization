import numpy as np
from ODEs.SIR import SIR
import pickle as pck
import pandas as pd
import matplotlib.pyplot as plt
def Autodiff_SIR(x, u, alpha, h, Ak, Bk):
    def dF_dx(x, u):
        return np.array([[-u*x[1], -u*x[0], 0],
                        [u*x[1], u*x[0]-alpha, 0],
                        [0, alpha, 0]])
    def dF_du(x):
        return np.array([-x[0]*x[1],
                         x[0]*x[1],
                         0])

    k1 = SIR(x, alpha, u)
    k2 = SIR(x + (h/2)*k1, alpha, u)
    k3 = SIR(x + h/2*k2, alpha, u)
    k4 = SIR(x + h*k3, alpha, u)

    xk_1 = x + h/6*(k1 + 2*k2 + 2*k3 + k4)

    k1_x = dF_dx(x, u)
    k2_x = dF_dx(x + h/2*k1, u)
    k3_x = dF_dx(x + h/2*k2, u)
    k4_x = dF_dx(x + h*k3, u)

    Cx = h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)

    k1_u = dF_du(x)
    k2_u = dF_du(x + h/2*k1)
    k3_u = dF_du(x + h/2*k2)
    k4_u = dF_du(x + h*k3)

    Cu = h/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)

    Ak_1 = Ak + np.matmul(Cx, Ak)
    Bk_1 = Bk + np.matmul(Cx , Bk) + Cu

    return xk_1, Ak_1,Bk_1


def Autodiff_SIR_Sensitivity(dt,start=0 , stop=0,plot=False, save = []):
    X0 = np.array([997, 3, 0])
    N = np.sum(X0)
    alpha = 1.0/3.0
    R0 = 10
    u = R0*alpha/N

    t = np.arange(start, stop,dt)


    A = [np.eye(3)]
    x = [X0]
    B = [np.array([0,0,0])]
    l = 0
    for i, h in enumerate(np.diff(t)):
        if (i % (0.1*len(t))) == 0:
            print('%d%%' %(l))
            l+=10
        xk_1, Ak_1, Bk_1 = Autodiff_SIR(x[-1], u, alpha, h, A[-1], B[-1])
        A.append(Ak_1)
        B.append(Bk_1)
        x.append(xk_1)

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

    df = pd.DataFrame(d, index = t)
    if np.any(save):
        df.to_pickle(save)
    return df