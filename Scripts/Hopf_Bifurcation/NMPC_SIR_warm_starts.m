clear all; close all; clc;

import casadi.*
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

CB = NlpCallback_forOCP('a', opti.nx, opti.ng);
opti.minimize(J);
Iter_max = 10;
opti.callback
% opt_maxit = struct('max_iter', Iter_max);
opti.solver('ipopt');%, opt_CB,opt_maxit);
Solver_starts_max = 10;

w_vec = zeros(length(w), Solver_starts_max);
w_vars = vertcat(w{:});
for i = 1:Solver_starts_max
    sol{i} = opti.solve_limited();
    w_vec(:,i) = sol{i}.value(w_vars);
    opti.set_initial(sol{i}.value_variables());
end



%% Plot
fig1 = figure()
hold on
legendstr = {};
for i = 1:Iter_max
    plot(t(1:end-1), w_vec(:,i)')
    legendstr{end+1} = num2str(i);
end
legend(legendstr);