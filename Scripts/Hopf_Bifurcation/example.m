# Control
u = MX.sym("u")

# State
x = MX.sym("x",3)
s = x[0] # position
v = x[1] # speed
m = x[2] # mass

# ODE right hand side
sdot = v
vdot = (u - 0.05 * v*v)/m
mdot = -0.1*u*u
xdot = vertcat(sdot,vdot,mdot)

# ODE right hand side function
f = Function('f', [x,u],[xdot])

# Integrate with Explicit Euler over 0.2 seconds
dt = 0.01  # Time step
xj = x
for j in range(20):
  fj = f(xj,u)
  xj += dt*fj

# Discrete time dynamics function
F = Function('F', [x,u],[xj,fj])

# Number of control segments
nu = 50 

# Control for all segments
U = MX.sym("U",nu) 
 
# Initial conditions
X0 = MX([0,0,1])

# Integrate over all intervals to generate a symbolic expression of the
# dynamics over time.
X=X0
for k in range(nu):
  X,dX = F(X,U[k])

# At this point, if we were to examine X, it would be the symbolic expression
# for the last state x_f.  To evaluate the actual value we need to wrap it in
# a function that accepts a given set of inputs U.
F_X = Function( 'F_X', [U], [X,dX] )

# Create some hard coded control input values over time that we'll use to generate
# a specific trajectory.
Us = np.ones( nu )

# Calcualte the trajectory taken based off our specified control inputs.
# This results in the final state x_f, but doesn't give us access to the
# trajectory along the way.
x_f,d_x_f = F_X( Us )

# Now instead reformulate the integration so that it, saves off the result 
# at each time step into a list so we can generate all the intermediate values 
# later for plotting.  Xs[i] will result in the 'i'th iteration as a symbolic
# expression.
X=X0
Xs = [X]
dXs = []
for k in range(1,nu):
  X,dX = F(Xs[k-1],U[k])
  Xs.append( X )
  dXs.append( dX )
    
# In order to evaluate each Xs, we need to create a wrapper function 
# around the symbolic expression that accepts our control input sequence U.
x_ks = []
dx_ks = []
for i,fx in enumerate(Xs[1:]):
    # Define the wrapper function around each symbolic expression 
    F_k = Function( 'F_k', [U], [fx,dXs[i]] )
    # Evaluate that symbolic expression with our given control inputs.
    # Note here that we pass in the entire sequence of 'nu' control inputs
    # but the symbolic expression will only use the ones it needs.  Only
    # the last x_k will use every element of the Us array.
    x_k,dx_k = F_k( Us )

    #print( x_k )
    
    # Save off the value of each x_k into a list.  We call full() here in order
    # to get the values as a plain array rather than as a DM class.  This will make
    # conversion to a numpy array possible later.
    x_ks.append( x_k.full() )
    dx_ks.append( dx_k.full() )

# Now convert this list of arrays into a 2d numpy array so we can plot it easily
x_ks = np.array( x_ks )
dx_ks = np.array( dx_ks )

# Plot the position and velocity.  (Could plot the mass too if we wanted to make a 3d plot.)
time = np.arange( 1,nu )
pos = x_ks[:,0]
vel = x_ks[:,1]
mass = x_ks[:,2]

sdot = dx_ks[:,0]
vdot = dx_ks[:,1]
mdot = dx_ks[:,2]

print( x_ks.shape )
print( time.shape )
print( pos.shape )
print( mdot.shape )

figure() 

plot( time, pos, label='position' )
plot( time, vel, label='velocity' )
plot( time, mass, label='mass' )

legend()

xlabel('time')

figure() 

plot( time, mass, label='mass' )

legend()

xlabel('time')


figure() 

plot( time, sdot, label='sdot' )
plot( time, vdot, label='vdot' )

legend()

xlabel('time')

figure() 

plot( time, mdot, label='mdot' )

legend()

xlabel('time')