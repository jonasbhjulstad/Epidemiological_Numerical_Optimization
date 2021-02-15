% Implementation of an interior point method 
clear variables
close all
clc

addpath('/Applications/casadiMatlab2020')
import casadi.*
% Problem definition
nv = 2;
x = MX.sym('x',nv);

% Equality contstraints
ne = 1;
g = sin(x(1)) - x(2)^2;
G = Function('G',{x},{g});
x_test = [2.,3.];
[g_test] = G(x_test);
disp(full(g_test))
g_test = -8.0907;

% Inequality contstraints
ni = 1;
h = x(1)^2 + x(2)^2 - 4;
H = Function('H',{x},{h});

% Objective
f = (x(1)- 4)^2 + (x(2) - 4)^2;
F = Function('F',{x},{f});

% Create CasADi object that evaluates the Jacobian of g
Jg = Function('Jg', {x},{jacobian(g,x)});

% Create CasADi object that evaluates the Jacobian of h
Jh = Function('Jh', {x},{jacobian(h,x)});

% Create CasADi object that evaluates the Jacobian of f
Jf = Function('Jf', {x},{jacobian(f,x)});

% Create CasADi object that evaluates the Hessian of the equalities
Hg = Function('Hg', {x},{hessian(g,x)});

% Create CasADi object that evaluates the Hessian of the inequalities
Hh = Function('Hh', {x},{hessian(h,x)});

% Create CasADi object that evaluates the Hessian of the Cost
Hf = Function('Hf', {x},{hessian(f,x)});

% Interior point solver
max_it = 100;
xk = [-2;4];

lk = 1*ones(ne,1);
vk = 1*ones(ni,1);
sk = 1*ones(ni,1);
iter = zeros(nv + ne + ni + ni,max_it);
iter(:,1) = [xk;lk;vk;sk];
tau = 1;
k_b = 1/3;
th_1 = 1.0e-8;
th_2 = 1.0e-8;
for i = 2:max_it
    %%% Build KKT system
    
    %Hessian
    Hf_e    = Hf(xk);
    Hg_e    = Hg(xk);
    Hh_e    = Hh(xk);
    Hl      = %COMPLETE;
    
    %Jacobians
    Jg_e    = Jg(xk);
    Jh_e    = Jh(xk);
    Jf_e    = Jf(xk);
    
    %Constraints
    g_e     = G(xk);
    h_e     = H(xk);
    
    KKT = %COMPLETE;
    
    lhs = %COMPLETE;
    
    % Termination condition
    if norm(lhs) < th_1
        if tau < th_2
            display('Solution found!')
            break;
        else
            tau = tau*k_b;
        end
    end
        
    sol = KKT\lhs;
    
    % line-serach
    max_ls = 100;
    x_step  = sol(1:nv);
    l_step  = sol(nv+1:nv+ne);
    v_step  = sol(nv+ne+1:nv+ne+ni);
    s_step  = sol(nv+ne+ni+1:end);
    alpha = 1;
    k_ls = 0.9;
    min_step = 1.0e-8;
    
    % COMPLETE: find step size that keeps s and v positive
    
    xk  = xk + alpha*x_step;
    lk  = lk + alpha*l_step;
    vk  = vk + alpha*v_step;
    sk  = sk + alpha*s_step;
    
    % Print some results
    display(['Iteration: ',num2str(i)]);
    display(['tau = ',num2str(tau)]);
    display(['norm(rhs) = ',num2str(norm(lhs))]);
    display(['step size = ',num2str(alpha)])
    iter(:,i) = [xk;lk;vk;sk];
end
iter = iter(:,1:i-1);
figure()
subplot(2,1,1)
plot(iter(1:2,:)')
grid on
xlabel('iterations')
ylabel('primal solution')
legend('x_1','x_2')
subplot(2,1,2)
plot(iter(3:4,:)')
grid on
xlabel('iterations')
ylabel('dual solution')
legend('\mu','\lambda')

% Plot feasible set, and iterations
figure()
pl = ezplot('sin(x) - y^2');
set(pl,'Color','red');
hold all
pl= ezplot('x^2 + y^2 - 4');
set(pl,'Color','blue');
ezcontour('(x- 4)^2 + (y - 4)^2')
plot(iter(1,:),iter(2,:),'--','Color','black')
plot(iter(1,:),iter(2,:),'o','Color','black')
title('Iterations in the primal space')
grid on


