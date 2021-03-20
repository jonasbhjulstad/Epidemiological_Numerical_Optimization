from casadi import *
# Declare model variables
from Parameters.ODE_initial import *


S = MX.sym('S')
I = MX.sym('I')
R = MX.sym('R')
x = vertcat(S, I, R)
lbd = MX.sym('lbd', 3)
u = MX.sym('u')
u_min = 0.01
u_max = 1
# Wu = N_pop**2/(k*(u_max-u_min)*2)
Wu = 1
beta = R0*alpha
# Model equations
N = 52
L = I ** 2 + Wu * u ** 2
xdot = vertcat(-beta * S * I / N_pop - u, beta * S * I / N_pop - alpha * I, alpha * I + u)
hamiltonian = L + lbd.T @ xdot
grad_h_u = Function('grad_h_u', [u, vertcat(x, lbd)], [jacobian(hamiltonian, u)])
grad_h = jacobian(hamiltonian, x).T
s_dot = Function('s_dot', [vertcat(x, lbd), u], [vertcat(xdot, -grad_h)])

x0 = [N_pop - I0, I0, 0]
# Objective term
# Formulate discrete time dynamics
# Fixed step Runge-Kutta 4 integrator
DT = T / N / M
h = DT *M #(Collocation step)
f = Function('f', [x, u], [xdot, L])
X0 = MX.sym('X0', 3)
U = MX.sym('U')
X = X0
Q = 0
X_plot = [X0]
u_lb = [u_min]
u_ub = [u_max]
u_init = [u_max]
u0 = u_max

sim_name = 'Vaccination'
