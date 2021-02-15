def RK4_Integrator(f, X, param, DT):
       k1 = f(X, param)
       k2 = f(X + DT / 2 * k1, param)
       k3 = f(X + DT / 2 * k2, param)
       k4 = f(X + DT * k3, param)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       return X
