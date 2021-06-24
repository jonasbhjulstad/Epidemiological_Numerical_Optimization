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
u_min = 1e-8*N_pop
u_max = 0.02*N_pop

# beta = R0*alpha
beta = R0*alpha
# Model equations
Wu = 1
L = I/N_pop + Wu * u/N_pop
xdot = vertcat(-beta * S * I / N_pop - u, beta * S * I / N_pop - alpha * I, alpha * I + u)
x0 = [N_pop - I0, I0, 0]
nx = len(x0)
# Formulate discrete time dynamics
# Fixed step Runge-Kutta 4 integrator
u0 = u_max
sim_name = 'Vaccination'

F = Function('f', [x, u], [xdot, L])
fk, Xk_plot, Qk_plot = RK4_M_times_plot(F, M, h)
f, X_plot, Q_plot = integrator_N_times_plot(fk, N, Xk_plot, Qk_plot)


