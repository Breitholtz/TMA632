from dolfin import *
import numpy as np
# Source term

#class Source(Expression):
#	def  eval(self, values, x):
#		dx=x[0]
#		values[0]=x[0]*sin(5*DOLFIN_PI*x[0])


# time stuff

T=2.0
num_steps=10
dt=T/num_steps

# Periodic BC's

class PeriodicBoundary(SubDomain):

	# 'left' boundary
	def inside(self, x, on_boundary):
		return bool(x[0]< DOLFIN_EPS or x[0]>-DOLFIN_EPS and on_boundary)

	# 'right' boundary, maps onto left boundary
	def map(self, x , y):
		y[0]=x[0]-2.0*DOLFIN_PI

# mesh and function space

mesh=IntervalMesh(10,0,2*DOLFIN_PI)
V = FunctionSpace(mesh,"CG",1,constrained_domain=PeriodicBoundary())

# def initial value
u_0= # what should we have here?
u_n=interpolate(u_0,V)


# variational problem
A=1
B=1
u=TrialFunction(V)
v=TestFunction(V)
f=Constant(0)     #Source term
a= A*inner(grad(u),grad(v))-B*inner(grad(u),v) 
L=f*v*dx

# time stepping
u=Function(V)
t=0

for n in range(num_steps):
	t+=dt

	# compute solution at time t
	solve(a==L,u) # bc's?

	u_n.assign(u)
	

file=File("Periodic.pvd")
file << u
