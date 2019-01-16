from dolfin import *
import matplotlib.pyplot as plt
import numpy as np




# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(near(x[0],0) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 2*DOLFIN_PI

def full_solution_method(T, num_steps, nx, u_0, f, B, D):

    set_log_level(40)
    point_start=Point(0)
    point_end=Point(2*DOLFIN_PI)
    dt=T/float(num_steps)

    mesh=IntervalMesh(nx, 0, 2*DOLFIN_PI)
    V=FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
    u=TrialFunction(V)
    v=TestFunction(V)

    u_n=project(u_0, V)

    a=D*u*v*dx+dt*grad(u)[0]*grad(v)[0]*dx+B*grad(u)[0]*v*dt*dx
    L=u_n*v*dx+f*v*dt*dx

    A_sys=assemble(a)


    u=Function(V)
    t=0

    for n in range(num_steps):
        t+=dt
        f.t=t
        solve(a==L, u)
        u_n.assign(u)

    return u_n

def first_order_split_solution(T, num_steps, nx, u_0, f, B, D):
    set_log_level(40)
    dt=T/float(num_steps)

    # Start and End Points of IntervalMesh
    point_start=Point(0.0)
    point_end=Point(2*DOLFIN_PI)
    mesh=IntervalMesh(nx, 0, 2*DOLFIN_PI)
    V=FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())
    u=TrialFunction(V)
    v=TestFunction(V)

    u_n=project(u_0, V)


    a_A=u*v*dx+dt*D*grad(u)[0]*grad(v)[0]*dx
    L_A=u_n*v*dx+dt*f*v*dx
    a_A_ass=assemble(a_A)

    a_B=u*v*dx+dt*B*grad(u)[0]*v*dx
    L_B=u_n*v*dx
    a_B_ass=assemble(a_B)

    u=Function(V)
    t=0

    for n in range(num_steps):
        f.t=t
        # Do system 2 update
        b_temp=assemble(L_A)
        solve(a_A_ass, u.vector(), b_temp)
        u_n.assign(u)

        # Do system 1 update
        b_temp=assemble(L_B)
        solve(a_B_ass, u.vector(), b_temp)
        u_n.assign(u)
        t+=dt


    return u_n



T=10
exp_end=2

num_steps_time=[np.power(2, k) for k in range(5, 15)]
mesh_sizes=[np.power(2,k) for k in range(4, 10)]


"""
## B non-unitary
#
u_0=Expression('exp(-t)*sin(x[0])', degree=4, t=0)
B=Expression('x[0]*sin(x[0])', degree=4)
f=Expression('exp(-t)*x[0]*sin(x[0])*cos(x[0])', degree=4, t=0)
u_true=Expression('exp(-t)*sin(x[0])', degree=4, t=10)
B_text="x*sinx"
D=1
"""


## B unitary
#
# Case B= (1)
u_0=Expression('exp(-t)*sin(x[0])', degree=4, t=0)
B=Constant(1.0)
f=Expression('exp(-t)*cos(x[0])', degree=4, t=0)
u_true=Expression('exp(-t)*sin(x[0])', degree=4, t=10)
D=1
B_text="1"

#(T, num_steps, nx, u_0, f, B, D):
## Iterate fine solutions
u_full_fine=full_solution_method(T,
                                 num_steps_time[-1],
                                 mesh_sizes[-1],
                                 u_0,
                                 f,
                                 B,
                                 D)
u_fos_fine=first_order_split_solution(T,
                                      num_steps_time[-1],
                                      mesh_sizes[-1],
                                      u_0,
                                      f,
                                      B,
                                      D)
params=[]
current_iteration=1

for i in range(len(num_steps_time)):
    for j in range(len(mesh_sizes)):
        print("Doing iteration %s out of %s"%(current_iteration, len(num_steps_time)*len(mesh_sizes)))
        #print("Running with params: (%s, %s)"%(num_steps_time[i], mesh_sizes[j]))
        u_full= full_solution_method(T,
                                     num_steps_time[i],
                                     mesh_sizes[j],
                                     u_0,
                                     f,
                                     B,
                                     D)
        l2_full_true=errornorm(u_true, u_full, 'L2')
        h1_full_true=errornorm(u_true, u_full, 'H1')
        l2_full_fine=errornorm(u_full_fine, u_full, 'L2')
        h1_full_fine=errornorm(u_full_fine, u_full, 'H1')
        
        u_fos = first_order_split_solution(T,
                                           num_steps_time[i],
                                           mesh_sizes[j],
                                           u_0,
                                           f,
                                           B,
                                           D)
        l2_fos_true=errornorm(u_true, u_fos, 'L2')
        h1_fos_true=errornorm(u_true, u_fos, 'H1')
        l2_fos_fine=errornorm(u_fos_fine, u_fos, 'L2')
        h1_fos_fine=errornorm(u_fos_fine, u_fos, 'H1')
        params.append([num_steps_time[i],
                       mesh_sizes[j],
                       l2_full_true,
                       h1_full_true,
                       l2_full_fine,
                       h1_full_fine,
                       l2_fos_true,
                       h1_fos_true,
                       l2_fos_fine,
                       h1_fos_fine])
                       
        current_iteration+=1

params=np.array(params)
np.save('./data/sols_results_1d_%s'%B_text, params)
