clear all; close all; clc;

import casadi.*

R0 = 1.5;
alpha = 0.1;
X = zeros(20, 3);
X0 = [500,10,0]';
beta = alpha*R0/sum(X0);
N = 10000;
t = linspace(0,1000,N);
DT = t(2)-t(1);



X(1,:) = X0;



J = 0;
for i = 1:N-1
        X(i+1,:) = RK4_Integrator(@(x,u)SIR(0, x, alpha, u), X(i,:)', beta, DT);
end


%% Plot


plot(t, X');
legend('S', 'I', 'R')