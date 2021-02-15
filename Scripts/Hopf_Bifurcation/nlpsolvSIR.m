clear all; close all; clc;

import casadi.*

R0 = 1.5;
alpha = 0.1;
X = zeros(20, 3);
X0 = [500,10,0]';
N_tot = sum(X0);
beta = alpha*R0/sum(X0);
N = 10;
t = linspace(0,100,N);
DT = t(2)-t(1);


w = {};
lbw = [];
ubw = [];
w0 = [];
Xk = X0;
J = 0;

g = [];
lbg = [];
ubg = [];
x = MX.sym('x', 3,1);
u = MX.sym('u');
xdot = [-u*x(1)*x(2);
            u*x(1)*x(2)-alpha*x(2);
            alpha*x(2)];
f = Function('f', {x, u}, {xdot});

Uk = MX.sym(['U_' num2str(1)]);

for k=0:N-1
    % New NLP variable for the control
    w = {w{:}, Uk};
    lbw = [lbw, 1e-4];
    ubw = [ubw,  1];
    w0 = [w0,  0.2];
    

    
    % Integration stage:
    k1 = f(Xk, Uk);
    k2 = f(Xk + DT/2 * k1, Uk);
    k3 = f(Xk + DT/2 * k2, Uk);
    k4 = f(Xk + DT * k3, Uk);
    Xk=Xk+DT/6*(k1 +2*k2 +2*k3 +k4);
    %Objective:
    J=J+Xk(2);
    % Add inequality constraint
     g = [g;Xk];
     lbg = [lbg; 1e-6*ones(3,1)];
     ubg = [ubg;  N*ones(3,1)];
    Uk = 0.0001;5
end

iteration_value_callback = SIR_iteration_values('SIR', []);

opts = struct('iteration_callback', iteration_value_callback, "ipopt.tol", 1e-8, "ipopt.max_iter", 50);
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob, opts);
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);
%% Plot

x = sol

% plot(t, X');
% legend('S', 'I', 'R')