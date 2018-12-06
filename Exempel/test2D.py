# Computing the 2D case of the convection-diffusion equation
#
#     u_t - Laplace(u) + dot(b,grad(u)) = f(x, y, t)
#
# on the square [0,2*PI]^2 with periodic boundary conditions.
# Initial value is u_0 = sin(x) + cos(y) and f = b_1(x, y)*e^(-t)*cos(x) - b_2(x, y)*e^(-t)*sin(y).

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

T = 5     # final time
dt = 0.1 # time step size
num_steps = int(T/dt) # number of time steps

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2*PI) and (2*PI, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and on_boundary)
		# and (not ((near(x[0], 0) and near(x[1], 2*DOLFIN_PI)) or (near(x[0], 2*DOLFIN_PI) and near(x[1], 0)))) --- Is this necessary?

    def map(self, x, y):
        if near(x[0], 2*DOLFIN_PI) and near(x[1], 2*DOLFIN_PI):
            y[0] = x[0] - 2*DOLFIN_PI
            y[1] = x[1] - 2*DOLFIN_PI
        elif near(x[0], 2*DOLFIN_PI):
            y[0] = x[0] - 2*DOLFIN_PI
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 2*DOLFIN_PI

# Create mesh and finite element
nx = ny = 64
mesh = RectangleMesh(Point(0, 0), Point(2*DOLFIN_PI, 2*DOLFIN_PI), nx, ny)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

u_0 = Expression('exp(-t)*(sin(x[0]) + cos(x[1]))', degree=2, t=0)
f = Expression('b_1*exp(-t)*cos(x[0]) - b_2*exp(-t)*sin(x[1])', degree=2, t=0, b_1=Constant(1), b_2=Constant(1))
u_n = project(u_0, V)

#f.b_1 = Expression('x[0]',degree=2)
#f.b_2 = Expression('x[1]',degree=2)
b = Constant((1,1))

a = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*dot(b,grad(u))*v*dx
L = (u_n + dt*f)*v*dx

# Create VTK file for saving solution
vtkfile = File('test/solution.pvd')

u = Function(V)
t = 0.0
for n in range(num_steps):

    # Update current time
    t += dt
    u_0.t = t
    f.t = t

    # Compute solution
    solve(a == L, u)

    # Save to file and plot solution
    vtkfile << (u, t)
    ua=u.vector().array()
    x = V.tabulate_dof_coordinates()
    i = np.argsort(x)
#    plt.plot(x[i],ua[i]);

    # Compute error at vertices
    u_e = project(u_0, V)
    error = np.abs(u_e.vector().array() - u.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Save plot
#plt.savefig("test.png")