from casadi import *
if __name__ == '__main__':
    x = SX.sym('x')
    z = SX.sym('z')
    p = SX.sym('p')
    dae = {'x': x, 'z': z, 'p': p, 'ode': z + p, 'alg': z * cos(z) - x}
    F = integrator('F', 'idas', dae)
    print(F)

    r = F(x0=0, z0=0, p=0.1, t0=1, tf=2)
    r['xf']
