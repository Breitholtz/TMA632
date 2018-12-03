# Computing the 1D case of the convection-diffusion equation
#
#     u_t - u_xx + u_x = f(x, t)
#
# on the interval [0,2*PI] with periodic boundary conditions.
# Initial value is u_0 = sin(x) and f = e^(-t)*cos(x).

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 2*DOLFIN_PI

# Create mesh and finite element
mesh = IntervalMesh(64, 0, 2*DOLFIN_PI)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

u_0 = Expression('exp(-t)*sin(x[0])', degree=2, t=0)
f = Expression('exp(-t)*cos(x[0])', degree=2, t=0)
u_n = interpolate(u_0, V)

a = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*u.dx(0)*v*dx
L = (u_n + dt*f)*v*dx

# Create VTK file for saving solution
vtkfile = File('test/solution.pvd')

u = Function(V)
t = 0
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
    u_e = interpolate(u_0, V)
    error = np.abs(u_e.vector().array() - u.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Save plot
plt.savefig("test.png")