# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg 2011
#
# First added:  2007-11-15
# Last changed: 2012-11-12
# Begin demo

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
        y[0] = x[0] - 1.0

# Create mesh and finite element
mesh = IntervalMesh(32, 0, 2*DOLFIN_PI)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

u_0 = Expression('exp(-t)*sin(x[0])', degree=2, t=0)
f = Expression('exp(-t)*cos(x[0])', degree=2, t=0)
u_n = interpolate(u_0, V)

a = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*u.dx(0)*v*dx
L = (u_n+ dt*f)*v*dx


# Create VTK file for saving solution
vtkfile = File('test/solution.pvd')

# Compute solution
#u = Function(V)
#solve(a == L, u)
# Time-stepping

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
    plot(u)

    # Compute error at vertices
    u_e = interpolate(u_0, V)
    error = np.abs(u_e.vector().array() - u.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

