class Singleshooter(object):
    def __init__(self, f, J,t_bounds, dt, x0, u, nx, NU):
        self.t_start = t_bounds[0]
        self.t_end = t_bounds[1]
        self.f = f
        self.J = J
        self.dt = dt
        self.nx = nx
        self.NU = NU
        self.x0 = x0
        self.u = u
        self.x_traj = [x0]
        self.q = 0
    def integrate(self):
        self.q = 0
        t = self.t_start
        xk = self.x0
        uk = self.u[0]
        perc = 1/self.NU
        u_ind = 0
        while t < self.t_end:
            if (t-self.t_start)/(self.t_end-self.t_start) > perc:
                perc += 1.0/(self.NU-1)
                u_ind +=1
                print(t)
                uk = self.u[u_ind]
            xk = self.f(xk, uk)
            self.x_traj.append(xk)
            self.q += self.J(xk, uk)
            t+= self.dt
        return xk