import numpy as np
T = 365.  # Time horizon
N = 52  # number of control intervals
M = 4  # RK4 steps per interval
DT = T / N / M
h = T/N #(Collocation step)

N_pop = 5.3e6
alpha = .1/9
# k = 100
R0 = 6.5
beta = R0*alpha
# Model equations
I0 = 500000
tgrid = np.arange(0, T, T/N)
tgrid_M = np.arange(0, T, T/N/M)

diff_tol = 1e-3
tau = 1
tau_tol = 1e-6
tau_factor = .6

d = 3