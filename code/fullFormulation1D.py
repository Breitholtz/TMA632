# Computing the 1D case of the convection-diffusion equation
#
#     u_t - u_xx + u_x = f(x, t)
#
# on the interval [0,2*PI] with periodic boundary conditions.
# Initial value is u_0 = sin(x) with source term f = b(x)*e^(-t)*cos(x).

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

T = 5 # final time
dt = 0.1 # time step size
num_steps = int(T/dt) # number of time steps

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(near(x[0],0) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 2*DOLFIN_PI

# Create mesh and finite element
nx = 64
mesh = IntervalMesh(nx, 0, 2*DOLFIN_PI)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

u_0 = Expression('exp(-t)*sin(x[0])', degree=4, t=0)
f = Expression('exp(-t)*cos(x[0])', degree=4, t=0)
u_n = project(u_0, V)

b = Expression('x[0]*sin(x[0])', degree=2)

a = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*b*grad(u)[0]*v*dx
L = (u_n + dt*dot(b,f))*v*dx

# Create VTK file for saving solution
vtkfile = File('full/solution1D.pvd')

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
    plt.plot(x[i],ua[i]);

    # Compute error at vertices
    u_e = project(u_0, V)
    error_max = np.abs(u_e.vector().array() - u.vector().array()).max()
    error_L2 = errornorm(u_0,u,'L2')
    error_H1 = errornorm(u_0,u,'H1')
    print('t = %.2f:\nError_max = %.3g\tError_L2 = %.3g\tError_H1 = %.3g' % (t, error_max, error_L2, error_H1))

    # Update previous solution
    u_n.assign(u)

# Save plot
plt.savefig("full/solution1D.png")