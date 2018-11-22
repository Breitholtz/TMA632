from fenics import *


# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)


def boundary(x, on_boundary):
	return on_boundary

class PeriodicBoundary(SubDomain):
	# left boundary
	def inside(self,x,on_boundary):
		return bool(x[0] < DOLFIN_EPS and x[0]> -DOLFIN_EPS and on_boundary)
	# right boundary
	def map(self,x,y):
		y[0]=x[0]-2*DOLFIN_PI
		y[1]=x[1]
pbc = PeriodicBoundary()
# mesh and space
L=DOLFIN_PI*2
SIZE=10
mesh = RectangleMesh(Point(0.0,0.0),Point(L, L),SIZE,SIZE,"right")
V=FunctionSpace(mesh, "CG",1,constrained_domain=pbc)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2 =', error_L2)
print('error_max =', error_max)

# Hold plot
interactive()