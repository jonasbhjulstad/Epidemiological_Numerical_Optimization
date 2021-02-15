from casadi import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ODE.Epidemiological import SIRD_discrete_t
from ODE.Epidemiological import SIRD
import datetime
from Integrator.RK4 import RK4_Integrator
import matplotlib
plt.style.use('seaborn-paper')
timevar_params = True
NB = 3
NG = 3
NM = 3

def B_t(t):
    if timevar_params:
        denoms = np.linspace(10,30,NB-1)
        #return np.array([[1] + [np.log(a*t) for a in denoms]])
        return np.array([[1] + [np.exp(-t/a) for a in denoms]])
    else:
        return np.array([1,1,1], ndmin=2)
def G_t(t):
    if timevar_params:
        return np.array([[1, t, t**2]])
    else:
        return np.array([1,1,1], ndmin=2)

def M_t(t):
    if timevar_params:
        denoms = np.linspace(10,30,NM-1)
        return np.array([[1] + [np.exp(-t/a) for a in denoms]])

    else:
        return np.array([1, 1, 1], ndmin=2)

if __name__ == '__main__':


    #Initial susceptible population in Veneto:
    # q_min = 0.0014



    q = MX.sym('q')
    gamma = MX.sym('gamma', 1, NG)
    nu = MX.sym('nu', 1, NM)
    beta = MX.sym('beta', 1, NB)

    #data_path = r'C:/Users/Jonas/Documents/covid19-ntnu/Italian_data/COVID-19/dati-regioni/'
    data_path = r'C:/Users/Jonas/Downloads/'

    #file = 'dpc-covid19-ita-regioni.csv'
    file = 'dpc-covid19-ita-andamento-nazionale.csv'
    df = pd.read_csv(data_path + file)
    #df = df[df['denominazione_regione'].str.contains('Veneto')]

    start_fit_date = '2020.02.24'
    end_fit_date = '2020.03.27'
    end_sim_date = '2020.05.27'

    dates = pd.to_datetime(df['data'])
    start_offset_fit = np.where(dates.dt.strftime('%Y.%m.%d') == start_fit_date)[0][0]
    end_offset_fit = np.where(dates.dt.strftime('%Y.%m.%d') == end_fit_date)[0][0] + 1
    start_offset_sim = start_offset_fit
    end_offset_sim = np.where(dates.dt.strftime('%Y.%m.%d') == end_sim_date)[0][0] + 1

    #Fit horizon, Sim horizon
    fh = [start_offset_fit, end_offset_fit]
    sh = [start_offset_sim, end_offset_sim]


    DT = np.diff(dates, axis=0)
    DT = [float(dt)/(3600.0e9*24.0) for dt in DT]
    t = [0.0]
    for dt in DT:
        t.append(t[-1] + dt)
    t_end = t[-1]



    I = df['totale_positivi'].values
    R = df['dimessi_guariti'].values
    D = df['deceduti'].values
    I_tot = df['totale_casi'].values
    I_fit = I[fh[0]:fh[1]]
    R_fit = R[fh[0]:fh[1]]
    D_fit = D[fh[0]:fh[1]]
    I_tot_fit = I_tot[fh[0]:fh[1]]
    dates_fit = dates[fh[0]:fh[1]]

    I_sim = I[sh[0]:sh[1]]
    R_sim = R[sh[0]:sh[1]]
    D_sim = D[sh[0]:sh[1]]
    I_tot_sim = I_tot[sh[0]:sh[1]]
    dates_sim = dates[sh[0]:sh[1]]

    N_pop = 4.906e6

    S_fit = [q*N_pop-I_fit[k]-R_fit[k]-D_fit[k] for k in range(len(D_fit))]
    t_fit = t[:fh[1]-fh[0]]
    DT_fit = DT[fh[0]:fh[1]]

    S_sim = [q*N_pop-I_sim[k]-R_sim[k]-D_sim[k] for k in range(len(D_sim))]
    t_sim = t[:sh[1]-sh[0]]
    DT_sim = DT[sh[0]:sh[1]]

    fig0, ax0 = plt.subplots(2)

    # ax.plot(dates, S, label='S')
    ax0[0].plot(dates, I, label='I')
    ax0[0].plot(dates, R, label='R')
    ax0[0].plot(dates, D, label='D')

    ax0[0].legend()

    ax0[1].plot(dates, I_tot, label='Total infected')
    ax0[1].legend()
    plt.show()

    X = np.array([S_fit, I_fit, R_fit, D_fit]).T
    X_diff = np.diff(X, axis=0)
    theta = horzcat(beta, gamma, nu).T
    Nw = theta.shape[0]
    J = 0
    #Forgetting factor:
    w = 0.7

    X0 = MX.sym('X0', 3)
    U = MX.sym('U', 1)
    Q = 0
    M = 30
    X_plot = [X0]
    for tk, dt, Xk, Delta_k in zip(t_fit,DT_fit, X, X_diff):

        Phi_k = SIRD_discrete_t(Xk, B_t(tk), G_t(tk), M_t(tk))
        J += w**(t_fit[-1]-tk)*(vertcat(*Delta_k)-Phi_k@theta)**2

    #Weights for the difference equations:
    w_x = DM([[.25], [.25], [.25], [0.25]])
    J = J.T @ w_x


    q_min = 0.0037#max(I + R + D)/N_pop
    q_max = 1
    q_init = q_min
    beta_min = [0.1]*NB
    beta_max = [10]*NB
    gamma_min = [0]*NG
    gamma_max = [10]*NG
    nu_min = [0]*NM
    nu_max = [1]*NM
    beta0 = [5]*NB
    gamma0 = [5]*NG
    nu0 = [0.1]*NM



    w_lb = beta_min + gamma_min + nu_min + [q_min]
    w_ub = beta_max + gamma_max + nu_max + [q_max]
    w0 = beta0 + gamma0 + nu0 + [q_init]


    w = vertcat(theta, q)

    nlp = {'x': w, 'f': J, 'g': []}

    filter_param = Function('filter_param', [w], [beta, gamma, nu, q])

    solver = nlpsol('id', 'ipopt', nlp)

    sol = solver(x0=w0, lbx=w_lb, ubx=w_ub)

    sol_beta, sol_gamma, sol_nu, sol_q = filter_param(sol['x'])
    sol_beta = sol_beta.full()

    S0 = sol_q.full()[0][0] * N_pop - I_fit[0] - R_fit[0] - D_fit[0]
    x0 = np.array([S0, I_fit[0], R_fit[0], D_fit[0]], ndmin=2).T

    X = [x0]
    param = {}
    theta_list = []
    # Simulate ODE:
    for tk, dt in zip(t_sim, DT_sim):
        theta_k = np.concatenate([B_t(tk) @ sol_beta.T,
                                G_t(tk) @ sol_gamma.T,
                                M_t(tk) @ sol_nu.T], axis=0)

        xdot = SIRD(X[-1], theta_k)

        X.append(X[-1] + xdot*dt)
        theta_list.append(theta_k)
    X = np.concatenate(X, axis=1).T
    theta_list = np.concatenate(theta_list, axis=1).T
    fig1, ax1 = plt.subplots(4)

    ax1[0].plot(dates_sim, I_sim, label='Infected data')
    ax1[0].plot(dates_sim, X[1:,1], label='Infected')
    ax1[1].plot(dates_sim, R_sim, label='Recovered data')
    ax1[1].plot(dates_sim, X[1:,2], label='Recovered')
    ax1[2].plot(dates_sim, D_sim, label='Deceased data')
    ax1[2].plot(dates_sim, X[1:,3], label='Deceased')
    _ = [ax1[3].plot(dates_sim,theta_list[:,i].T, label=l) for i, l in enumerate([r'$\beta$', r'$\gamma$', r'$\nu$'])]
    _ = [x.legend() for x in ax1]


    plt.show()


