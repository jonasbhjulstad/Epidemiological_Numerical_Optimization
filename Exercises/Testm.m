import casadi.*


x = MX.sym('x',2); % Two states

% Expression for ODE right-hand side
z = 1-x(2)^2;
rhs = [z*x(1)-x(2);x(1)];

ode = struct;    % ODE declaration
ode.x   = x;     % states
ode.ode = rhs;   % right-hand side

% Construct a Function that integrates over 4s
F = integrator('F','cvodes',ode,struct('tf',4));

% Start from x=[0;1]
res = F('x0',[0;1]);

disp(res.xf)

% Sensitivity wrt initial state
res = F('x0',x);
S = Function('S',{x},{jacobian(res.xf,x)});

disp(S([0;1]))