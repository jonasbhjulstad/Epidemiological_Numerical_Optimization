clear all; close all; clc;

import casadi.*
DT = 0.1;
U = 0.5;
alpha = 0.4;
beta = 0.0001;
N = 100;
t = linspace(0,20,N);
DT = t(2)-t(1);

X = zeros(20, 3);
X0 = [10000,100,0]';

opti = Opti();
Xvar = opti.variable(N-1, 3);

% opti.set_initial(Xvar, repmat(X0', N-1, 1));
for k = 1:3
    opti.subject_to(Xvar(:,k) >= 0);
    opti.subject_to(Xvar(:,k) <= sum(X0));
end
U = opti.variable(N-1);
opti.subject_to(0.001<= U <= 1);
% opti.set_initial(U, 0.3*ones(N-1,1));
J = 0;
for i = 1:N-1
    if(i == 1)
        Xk = RK4_Integrator(@(x,u)SIR(0, x, alpha, u), X0, U(i), DT);
    else
        Xk = RK4_Integrator(@(x,u)SIR(0, x, alpha, u), Xvar(i,:)', 0.1, DT);
    end
    opti.subject_to(Xvar(i,:)' == Xk);
    opti.subject_to(sum(Xvar(i,:)) == sum(X0));
    J = J + Xk(2)^2;
end
opti.minimize(J);
opti.solver('ipopt');
sol = opti.solve()
%% Plot
x = sol.value(Xvar);
u = sol.value(U);
plot(t, [X0'; x]');
legend('S', 'I', 'R')