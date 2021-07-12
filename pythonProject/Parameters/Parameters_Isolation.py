# from casadi import *
# # Declare model variables
# import sys
# from os.path import dirname, abspath
# parent = dirname(dirname(abspath(__file__)))
# sys.path.append(parent)
# from Parameters.ODE_initial import *


from casadi import *
# Declare model variables
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Parameters.ODE_initial import *
from RK4.Integrator import RK4_M_times_plot, integrator_N_times_plot
S = MX.sym('S')
I = MX.sym('I')
R = MX.sym('R')
x = vertcat(S, I, R)
u = MX.sym('u')
u_min = 1e-4
u_max = .2
# beta = R0*alpha
beta = R0*alpha
# Model equations
Wu = 1
L = I/N_pop + Wu * u
xdot = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I - u*I, alpha * I + u*I)


x0 = [N_pop - I0, I0, 0]
nx = len(x0)
# Formulate discrete time dynamics
# Fixed step Runge-Kutta 4 integrator
u0 = u_max
sim_name = 'Social_Distancing'

#Functions for computing states and derivatives + plot values
F = Function('f', [x, u], [xdot, L])
fk, Xk_plot, Qk_plot = RK4_M_times_plot(F, M, h)
f, X_plot, Q_plot = integrator_N_times_plot(fk, N, Xk_plot, Qk_plot)





# S = MX.sym('S')
# I = MX.sym('I')
# R = MX.sym('R')
# x = vertcat(S, I, R)
# lbd = MX.sym('lbd', 3)
# u = MX.sym('u')
# u_min = 0
# u_max = .2
# beta = R0*alpha
# # Model equations

# xdot = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I - u, alpha * I + u)

# grad_h = -vertcat((lbd[1] - lbd[0])*beta*I/N_pop, (lbd[1] - lbd[0])*beta*S/N_pop - lbd[1]*alpha + lbd[2], 0)
# grad_h_u = Function('grad_h_u', [u, vertcat(x, lbd)], [jacobian(hamiltonian, u)])

# s_dot = Function('s_dot', [vertcat(x, lbd), u], [vertcat(xdot, -grad_h)])
# x0 = [N_pop - I0, I0, 0]
# # Objective term
# Wu = .1
# L = I + Wu * (u - u_max)**2/(u_max-u_min)**2

# # Formulate discrete time dynamics
# # Fixed step Runge-Kutta 4 integrator
# f = Function('f', [x, u], [xdot, L])
# X0 = MX.sym('X0', 3)
# U = MX.sym('U')
# X = X0
# Q = 0
# X_plot = [X0]
# u_lb = [u_min]
# u_ub = [u_max]
# u_init = [u_max]
# u0 = u_max

# sim_name = 'Isolation'


# lbd = MX.sym('lbd', 3)
# xdot_PMP = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I - u*I, u*S + alpha*I)

# hamiltonian = I**2 + Wu*u**2 + lbd[0]*(-u*S - S*I*beta/N_pop) + lbd[1]*(S*I*beta/N_pop - alpha*I) + lbd[2]*(alpha*I + u*S)
# grad_H = jacobian(hamiltonian, vertcat(S, I, R))
# F = Function('F', [vertcat(S, I,R, lbd), u], [vertcat(xdot_PMP, -grad_H.T)])


