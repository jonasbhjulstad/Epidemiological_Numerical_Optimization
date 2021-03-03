from casadi import *
T = 28.  # Time horizon
N = 50  # number of control intervals
N_pop = 5.3e6
alpha = .1/9
# Declare model variables
S = MX.sym('S')
I = MX.sym('I')
R = MX.sym('R')
x = vertcat(S, I, R)
lbd = MX.sym('lbd', 3)
u = MX.sym('u')
u_min = 0
u_max = .2
k = 100
Wu = N_pop**2/(k*(u_max-u_min)*2)
R0 = 5
beta = R0*alpha
# Model equations
xdot = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I - u*I, alpha * I + u * I)
hamiltonian = I**2 + Wu*u**2 + lbd[0]*(-u*S - S*I*beta/N_pop) + lbd[1]*(S*I*beta/N_pop - alpha*I)
I0 = 20000
x0 = [N_pop - I0, I0, 0]
# Objective term
L = I ** 2 + Wu * u ** 2

# Formulate discrete time dynamics
# Fixed step Runge-Kutta 4 integrator
M = 30  # RK4 steps per interval
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

sim_name = 'Isolation'


lbd = MX.sym('lbd', 3)
xdot_PMP = vertcat(-beta * S * I / N_pop, beta * S * I / N_pop - alpha * I - u*I, u*S + alpha*I)

hamiltonian = I**2 + Wu*u**2 + lbd[0]*(-u*S - S*I*beta/N_pop) + lbd[1]*(S*I*beta/N_pop - alpha*I) + lbd[2]*(alpha*I + u*S)
grad_H = jacobian(hamiltonian, vertcat(S, I, R))
F = Function('F', [vertcat(S, I,R, lbd), u], [vertcat(xdot_PMP, -grad_H.T)])

