from fenics import *
import numpy as np
import matplotlib.pyplot as plt



# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 2*DOLFIN_PI
            y[1] = x[1] - 2*DOLFIN_PI
        elif near(x[0], 1):
            y[0] = x[0] - 2*DOLFIN_PI
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 2*DOLFIN_PI




T = 10.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size


error_l2=[]
error_h1=[]
error_vertex=[]
timestamp_error_l2=[]
timestamp_error_h1=[]
timestamp_vertex_error=[]

# Create mesh and define function space
nx = ny = 64
#mesh = UnitSquareMesh(nx, ny)

#Start and End points of RectangleMesh
point_start=Point((0,0))
point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))

mesh = RectangleMesh(point_start, point_end, nx, ny)
pbc=PeriodicBoundary()
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)

# Define boundary condition
u_0 = Expression('sin(x[0])+cos(x[1])',
                 degree=2)

# True solution
u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                  t=0,
                  degree=2)


# Define initial value
u_n = project(u_0, V)
u = TrialFunction(V)
v = TestFunction(V)





B_1=1
B_2=1
B=Constant((B_1, B_2))
D=Constant(1.0)

f=Expression('exp(-t)*(2.0*sin(x[0])+2.0*cos(x[1])+B_1*cos(x[0])+B_2*sin(x[1]))',
                  B_1=Constant(B_1), B_2=Constant(B_2), t=0, degree=2)

a=D*u*v*dx+dt*dot(grad(u), grad(v))*dx+dot(B, grad(u))*v*dt*dx
L=u_n*v*dx+f*v*dt*dx
A_sys=assemble(a)

u=Function(V)
t=0

# Append error at start, and start time
error_l2.append(errornorm(u_true, u_n, 'L2'))
timestamp_error_l2.append(t)
error_h1.append(errornorm(u_true, u_n, 'H1'))
timestamp_error_h1.append(t)
vertex_values_u_true=u_true.compute_vertex_values(mesh)
vertex_values_u_n=u_n.compute_vertex_values(mesh)
error_max=np.max(np.abs(vertex_values_u_true-vertex_values_u_n))
error_vertex.append(error_max)
timestamp_vertex_error.append(t)

for n in range(num_steps):

    t+=dt
    f.t=t

    b_temp=assemble(L)
    solve(A_sys, u.vector(), b_temp)
    u_n.assign(u)

    u_true.t=t

    # Calulate L2 Error norm for this time step
    error_l2.append(errornorm(u_true, u_n, 'L2'))
    timestamp_error_l2.append(t)
    error_h1.append(errornorm(u_true, u_n, 'H1'))
    timestamp_error_h1.append(t)

    vertex_values_u_true=u_true.compute_vertex_values(mesh)
    vertex_values_u_n=u_n.compute_vertex_values(mesh)
    error_max=np.max(np.abs(vertex_values_u_true-vertex_values_u_n))
    error_vertex.append(error_max)
    timestamp_vertex_error.append(t)
    
    p=plot(u)
    p.set_cmap("viridis")

    plt.colorbar(p)

    plt.savefig('./images/solution_%s.png'%t)
    plt.subplots()

    u_n.assign(u)

# Plot error over time
plt.subplot(2,2,1)
plt.plot(timestamp_error_l2, error_l2, 'rx')
plt.title("L2 Error Norm over Time")
plt.xlabel("Time")
plt.ylabel("L2 Error Norm")

plt.subplot(2,2,2)
plt.plot(timestamp_error_h1, error_h1, 'bx')
plt.title("H1 Error Norm over Time")
plt.xlabel("Time")
plt.ylabel("H1 Error Norm")

plt.subplot(2,2,3)
plt.plot(timestamp_vertex_error, error_vertex, 'kx')
plt.title("Max Vertex Error over time")
plt.xlabel("Time")
plt.ylabel("Max Vertex Error")




plt.savefig('./images/error_l2_h1.png')
