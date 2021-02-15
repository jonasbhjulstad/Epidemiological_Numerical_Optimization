addpath('/Applications/casadiMatlab2020')
import casadi.*
clear
clc
% We want to model a chain attached to two supports and hanging in between. Let us discretise
% it with N mass points connected by N-1 springs. Each mass i has position (yi,zi), i=1,...,N.
% The equilibrium point of the system minimises the potential energy.
% The potential energy of each spring is
% Vi=D_i/2 * ((y_i-y_{i+1})^2 + (z_i-z_{i+1})^2)
% The gravitational potential energy of each mass is
% Vg_i = m_i*g0*z_i
% The total potential energy is thus given by:
 
% Vchain(y,z) = 1/2*sum{i=1,...,N-1} D_i ((y_i-y_{i+1})^2+(z_i-z_{i+1})^2) + g0 * sum{i=1,...,N} m_i * z_i
% where y=[y_1,...,y_N] and z=[z_1,...,z_N]
% We wish to solve
% minimize{y,z} Vchain(y, z)
% Subject to the piecewise linear ground constraints:
% z_i >= zin
% z_i - 0.1*y_i >= 0.5

% Constants
N = 40;
m_i = 40.0/N;
D_i = 70.0*N;
g0 = 9.81;

% Objective function
Vchain = 0;

% Variables
x = {};

% Variable bounds
lbx = [];
ubx = [];

% Constraints
g = [];
% Constraint bounds
lbg = [];
ubg = [];
% Loop over all chain elements
for i=0:N
   % Previous point
   if i>0
      y_prev = y_i;
      z_prev = z_i;
   end
   
   % Create variables for the (y_i, z_i) coordinates
   y_i = SX.sym(['y_', num2str(i)]);
   z_i = SX.sym(['z_', num2str(i)]);

   % Add to the list of variables
   x = [x, {y_i, z_i}];

   if (i==0)
    lbx =  [lbx;-2.; 1.];
    ubx =  [ubx;-2.; 1.];
   elseif (i==N)
    lbx = [lbx; 2.; 1.];
    ubx = [ubx; 2.; 1.];
   else
    lbx = [lbx;-inf; -inf];
    ubx = [ubx; inf;  inf];
   end
   if (i >= 2) && (i <= (N-2))
    %g = [g; z_i-0.5; z_i-0.1*y_i-0.5];
    g = [g; z_i-0.5 + 0.1*y_i^2];
   end
   % Spring potential
   if (i>0)
      Vchain = Vchain + D_i/2*((y_prev-y_i)^2 + (z_prev-z_i)^2);
   end
   
   % Graviational potential
   Vchain =  Vchain + g0 * m_i * z_i;

end

% Formulate QP
qp = struct('x',vertcat(x{:}),'f',Vchain,'g',g);

% Solve with IPOPT
solver = qpsol('solver', 'qpoases', qp);

% Get the optimal solution
arg = struct('lbx',lbx, 'ubx',ubx, 'lbg',lbg, 'ubg',ubg);
sol = solver.call(arg);

% Retrieve the result
x_opt = full(sol.x);
Y0 = x_opt(1:2:end);
Z0 = x_opt(2:2:end);
 
% Plot the result
plot(Y0,Z0,'o-')
ys = linspace(0.,2.,100);
hold all;
xlabel('y [m]')
ylabel('z [m]')
title('Hanging chain QP')
grid on
ylim([0.3,1 ])