clear all; close all; clc;

import casadi.*

datafile = 'data\NMPC_SIR.dat';
writematrix([],'NMPC_SIR.dat');
N = 100;
t = linspace(0,20, N);
X0 = [10000,10,0]';
opti = Opti();
Xk = X0;
N_tot = sum(Xk);

R0 = 0.8;
alpha = 0.5;
beta = R0*alpha/N_tot;

J = 0;
w_vec = {};
x = MX.sym('x', 3,1);
u = MX.sym('u');
xdot = [-u*x(1)*x(2);
            u*x(1)*x(2)-alpha*x(2);
            alpha*x(2)];
f = Function('f', {x, u}, {xdot});

X = opti.variable(N, 3);
Nu = 3;
Uk = opti.variable();
opti.subject_to(1e-8 <= Uk <= 0.01);
opti.set_initial(Uk, 1e-8);
w = {Uk};
DT = t(2)-t(1);
X(2,:) = RK4_Integrator(f, X0, Uk, DT);
for i = 2:N-1
    if( mod(i,((N-1)/Nu)) == 0)
        Uk = opti.variable();
        opti.subject_to(1e-8 <= Uk <= 0.1);
        opti.set_initial(Uk, 1e-8);
    end
    DT = t(i+1)-t(i);
    opti.subject_to(X(i+1,:)' == RK4_Integrator(f, X(i,:)', Uk, DT));
    J = J + X(i+1,2)^2 + 1e-1*(1/Uk)^2;
    w = {w{:}, Uk};
end

opti.minimize(J);
opti.callback(@(i) writecell(opti.debug.value_variables(),datafile, 'WriteMode', 'append'));
opti.solver('ipopt');%

sol = opti.solve()






%% Plot
fig1 = figure()
hold on

legend(legendstr);