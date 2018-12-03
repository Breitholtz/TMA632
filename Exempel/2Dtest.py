# a new try to get all this to actually work

from __future__ import print_function
from fenics import *
import numpy as np

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# define periodic bc's
class PeriodicBoundary(SubDomain):
	
	#left boundary
	def inside(self,x,on_boundary):
		return bool(x[0]< DOLFIN_EPS and x[0]>-DOLFIN_EPS and on_boundary)

	# map right boundary onto left
	def map(self,x,y):
		y[0]=x[0]-1.0
		y[1]=x[1]


# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())



# Define dirichlet boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(Constant(1), V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
B= Expression("x[0]","0",degree=3) # velocity field for convection

F = dt*dot(B,grad(u))*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx 
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

  

    # Update previous solution
    u_n.assign(u)

print("all done!" )
file=File("periodictest.pvd")
file << u 
