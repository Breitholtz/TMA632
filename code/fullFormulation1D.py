# Computing the 1D case of the convection-diffusion equation
#
#     u_t - u_xx + u_x = f(x, t)
#
# on the interval [0,2*PI] with periodic boundary conditions.
# Initial value is u_0 = sin(x) with source term f = b(x)*e^(-t)*cos(x).

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def computation(meshpoints,timestep):
 T = 5 # final time
 dt = timestep # time step size
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
 nx = meshpoints
 mesh = IntervalMesh(nx, 0, 2*DOLFIN_PI)
 V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
 
 # Define variational problem
 u = TrialFunction(V)
 v = TestFunction(V)

 u_0 = Expression('exp(-t)*sin(x[0])', degree=4, t=0)
 f = Expression('exp(-t)*cos(x[0])', degree=4, t=0)
 u_n = project(u_0, V)

 b  = Expression('x[0]*sin(x[0])', degree=2)

 a = u*v*dx + dt*dot(grad(u), grad(v))*dx + dt*b*grad(u)[0]*v*dx
 L = (u_n + dt*dot(b,f))*v*dx

 # Create VTK file for saving solution
 vtkfile = File('1Dfull/solution1D.pvd')

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
    ua=u.vector().get_local()
    x = V.tabulate_dof_coordinates().reshape(nx)
    i = np.argsort(x)
    plt.plot(x[i],ua[i]);

    # Compute error at vertices
    u_e = project(u_0, V)
    error_max = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    error_L2 = errornorm(u_0,u,'L2')
    error_H1 = errornorm(u_0,u,'H1')
    print('t = %.2f:\nError_max = %.3g\tError_L2 = %.3g\tError_H1 = %.3g' % (t, error_max, error_L2, error_H1))

    # Update previous solution
    u_n.assign(u)

 # Save plot
 plt.savefig("1Dfull/solution1D.png")
 return (error_L2, error_H1, mesh.hmax(),dt)
def main(dt):
	H=8 # amount we want to multiply the #meshpoints with
	dt=10 # amount we want to decrease the timestep by dividing
	L2err=[]
	H1err=[]
	h=[]
	DT=[]
	# loop over several meshsizes/timesteps
	for i in range(1,10):
		if(dt):
			L2,H1,hmax,timestep = computation(64,1/(10*i*dt))
		else:
			L2,H1,hmax,timestep = computation(i*H,0.1)
		L2err.append(L2)
		H1err.append(H1)
		h.append(hmax)
		DT.append(timestep)
	print("Meshsize")
	print(h)
	print("L2 error")
	print(L2err)
	print("H1 error")
	print(H1err)
	print("Timestep")
	print(DT)
main(1)
