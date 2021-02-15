alpha = 0.1;
beta = 0.0001;
t = [0,40];

x0 = [10000,100,0]';

[T, X] = ode45(@(t,x)SIR(t,x,alpha, beta), t, x0);

plot(T, X)