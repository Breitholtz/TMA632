# a new try to get all this to actually work

from __future__ import print_function
from fenics import *
import numpy as np

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size

# define periodic bc's
class PeriodicBoundary(SubDomain):
	
	#left boundary
	def inside(self,x,on_boundary):
		return bool(x[0]< DOLFIN_EPS and x[0]>-DOLFIN_EPS and on_boundary)

	# map right boundary onto left
	def map(self,x,y):
		y[0]=x[0]-1.0
		y[1]=x[1]

	# TODO: up maps to down
# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())



# Define initial boundary condition
u_0 = Expression('exp(t)*sin(x[0])*cos(x[1])',
                 degree=2, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_0, boundary)

# Define initial value as interpolation of DBC
u_n = interpolate(u_0, V)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
B= Constant(tuple((1,1))) # velocity field

F = u*v*dx + dt*dot(B,grad(u))*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx 
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_0.t = t

    # Compute solution
    solve(a == L, u, bc) 

    # Update previous solution
    u_n.assign(u)
	# TODO save solutions for every timestep
print("all done!" )
file=File("periodictest.pvd")
file << u 

# TODO: compute error against known function in L2 and H1 norms

