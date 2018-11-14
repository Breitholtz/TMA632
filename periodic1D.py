from dolfin import *

# Source term

class Source(Expression):
	def  eval(self, values, x):
		dx=x[0]
		values[0]=x[0]*sin(5*DOLFIN_PI*x[0])


# Periodic BC's

class PeriodicBoundary(SubDomain):

	# 'left' boundary
	def inside(self, x, on_boundary):
		return bool(x[0]< DOLFIN_EPS or x[0]>-DOLFIN_EPS and on_boundary)

	# 'right' boundary, maps onto left boundary
	def map(self, x , y):
		y[0]=x[0]-2.0*DOLFIN_PI

# mesh and function space

mesh=IntervalMesh(100,0,2*DOLFIN_PI)
V = FunctionSpace(mesh,"CG",1,constrained_domain=PeriodicBoundary())

# variational problem

u=TrialFunction(V)
v=TestFunction(V)
f=Source()
a=#TODO
L=f*v*dx

# compute solution
u=Function(V)
solve(a==L,u)

# save solution
file=File("Periodic.pvd")
file << u
