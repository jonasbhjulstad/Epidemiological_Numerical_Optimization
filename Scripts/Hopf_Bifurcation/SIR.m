function xdot = SIR(x, alpha, beta)
S = x(1);
I = x(2);
R = x(3);

S_dot = -beta*S*I;
I_dot = beta*S*I - alpha*I;
R_dot = alpha*I;
xdot = [S_dot, I_dot, R_dot]';
end