import numpy as np
import pandas as pd
from casadi import *
def SIR(x, beta,param):
    alpha = param['alpha']

    S = x[0]
    I = x[1]
    R = x[2]

    S_dot = -beta*S*I
    I_dot = beta*S*I - alpha*I
    R_dot = alpha*I
    return np.array([S_dot, I_dot, R_dot])

def SIRD(x, theta):


    S = x[0][0]
    I = x[1][0]
    R = x[2][0]
    D = x[3][0]

    SI_dyn = S*I/(S + I)

    Phi = [[-SI_dyn, 0, 0],
                    [SI_dyn, -I, -I],
                    [0, I, 0],
                    [0, 0, I]]

    return Phi @ theta

def SIRD_parameters(type='expected'):
    df = pd.read_pickle(r'../data/SIRD_Brazil.pck')
    params = df['Values'][0:4].values
    if(type == 'expected'):
        ret_ind = 0
    elif(type == 'upper'):
        ret_ind = 2
    else:
        ret_ind = 1

    R0 = params[3][ret_ind]
    beta = params[0][ret_ind]
    gamma = params[1][ret_ind]
    rho = params[2][ret_ind]

    return {'beta': beta, 'gamma':gamma, 'rho': rho, 'R0': R0}


def SIRD_discrete(x, beta, gamma, nu):
    beta_term = -x[0]*x[1]/(x[0]+x[1])
    A = np.array([[-beta_term, 0, 0],
                  [beta_term, -x[1], -x[1]],
                  [0, x[1], 0],
                 [0, 0, x[1]]])

    return A @ np.reshape([beta, gamma, nu], (-1, 1))

def SIRD_discrete_t(x, B, G, M):

    beta_term = x[0]*x[1]/(x[0]+x[1])*B
    col0_zeros = DM.zeros(1,beta_term.shape[1])

    gamma_term = x[1]*G
    col1_zeros = DM.zeros(1,gamma_term.shape[1])

    nu_term = x[1]*M
    col2_zeros = DM.zeros(1,nu_term.shape[1])

    Phi = vertcat(
                horzcat(-beta_term, col1_zeros, col2_zeros),
                horzcat(beta_term, -gamma_term, -nu_term),
                horzcat(col0_zeros, gamma_term, col2_zeros),
                horzcat(col0_zeros, col1_zeros, nu_term)
                )


    return Phi

