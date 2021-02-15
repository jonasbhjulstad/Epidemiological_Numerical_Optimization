
x0 = 1;
JH = @(x, lbd) 2*x -lbd*cos(x);
lbd_0 = 1;
F = @(x,u) u-sin(x);
tol = 1e-5;
t = linspace(0,4,100);
X = PMP_solve(x0,lbd_0, tol, F, JH,t)
function [X] = PMP_solve(x0,lbd_0, tol, F, JH, t)
    u = -lbd_0;
    r = 10000;
    N = 10;
    lbd = zeros(N,1);
    lbd(1) = lbd_0;
    n = 1;
    X = cell(N,1);
   
    while (abs(r) > tol) || (n > N)
        x_dot = @(x)F(x, u);
        lbd_dot = @(x,lbd)-JH(x,lbd);
        
        w = @(x,lbd)[x_dot(x);lbd_dot(x,lbd)];
        
        X{n} = ode45(w, t, [x0;lbd_0(n)]);
        lbd_f(n) = X{n}(end,2);
        
        r(n) = lbd_f(n);
        del_lbd(n) = lbd_f(n)-lbd_0(n);
        lbd_0(n+1) = lbd_0(n) - r/del_lbd(n);
        n = n+1;
    end
end
        