import numpy as np
def SIR(x, alpha,N, R0):
    S = x[0]
    I = x[1]
    R = x[2]
    beta = R0*alpha

    S_dot = -beta*S*I/N
    I_dot = beta*S*I/N - alpha*I
    R_dot = alpha*I
    return np.array([S_dot, I_dot, R_dot])

def SIR_odesol(t, x,param):
    alpha = param[0]
    R0 = param[1]
    N = param[2]
    beta = R0*alpha
    S = x[0]
    I = x[1]
    R = x[2]

    S_dot = -beta*S*I/N
    I_dot = beta*S*I/N - alpha*I
    R_dot = alpha*I
    return np.array([S_dot, I_dot, R_dot])

def SIR_linearized(x, J):
    alpha = param[0]
    R0 = param[1]
    N = param[2]
    beta = R0*alpha
    S = x[0]
    I = x[1]
    R = x[2]

    return J @ np.array([S, I, R])