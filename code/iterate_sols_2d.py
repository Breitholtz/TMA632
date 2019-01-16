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
                        (near(x[0], 2*DOLFIN_PI) and near(x[1], 2*DOLFIN_PI)))) and on_boundary)

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

def full_system_solution(T,num_steps, nx, ny, u_0, B, f, D):

    set_log_level(50)
    # Start and End Points of RectangleMesh
    point_start=Point((0,0))
    point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))
    dt=T/float(num_steps)

    mesh = RectangleMesh(point_start, point_end, nx, ny)
    pbc=PeriodicBoundary()
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)


    # Define initial value
    u_n = project(u_0, V)
    u = TrialFunction(V)
    v = TestFunction(V)


    a=D*u*v*dx+dt*dot(grad(u), grad(v))*dx+dot(B, grad(u))*v*dt*dx
    L=u_n*v*dx+f*v*dt*dx
    A_sys=assemble(a)

    u=Function(V)
    t=0

    f.t=t
    for n in range(num_steps):
        t+=dt
        f.t=t
        solve(a==L,u)
        u_n.assign(u)


    return u_n
    
def first_order_split_solution(T,num_steps, nx, ny, u_0, B, f, D):
    set_log_level(50)
    dt=T/float(num_steps)

    # Start and End Points of RectangleMesh
    point_start=Point((0,0))
    point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))

    mesh = RectangleMesh(point_start, point_end, nx, ny)
    pbc=PeriodicBoundary()
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)


    # Define initial value
    u_n = project(u_0, V)
    u = TrialFunction(V)
    v = TestFunction(V)



    a_A=u*v*dx+dt*D*dot(grad(u), grad(v))*dx
    L_A=u_n*v*dx+dt*f*v*dx
    a_A_ass=assemble(a_A)

    a_B=u*v*dx+dt*dot(B, grad(u))*v*dx
    L_B=u_n*v*dx
    a_B_ass=assemble(a_B)

    u=Function(V)
    t=0

    f.t=t
    for n in range(num_steps):
        # Do system 2 update

        f.t=t
        b_temp=assemble(L_A) 
        solve(a_A_ass, u.vector(), b_temp)
        u_n.assign(u)
    
        # Do system 1 update

        b_temp=assemble(L_B)
        solve(a_B_ass, u.vector(), b_temp)
        u_n.assign(u)
        t+=dt
        u_true.t=t

    return u_n


T=10
exp_end=2
num_steps_time=[np.power(2, k) for k in range(5, 12)] #13
mesh_sizes=[np.power(2,k) for k in range(3, 8)] #8


"""
## B non-unitary
#
u_0=Expression('sin(x[0])+cos(x[1])', degree=4)
u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                  t=10, degree=4)
B=Expression(('x[0]*x[0]*sin(x[1])',
              'x[1]*x[1]*cos(x[0])'),
             degree=4)
D=Constant(1.0)
f=Expression('exp(-t)*sin(x[1])*cos(x[0])*(x[0]*x[0]-x[1]*x[1])', degree=4, t=0)
B_text="(x*x*sin(y), y*y*cos(x))"
"""

# B Unitary
#
u_0=Expression('sin(x[0])+cos(x[1])', degree=4)
u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                  t=10, degree=4)
B=Constant((1.0,1.0))
D=Constant(1.0)
f=Expression('exp(-t)*(cos(x[0])-sin(x[1]))', degree=4, t=0)
B_text="(1,1)"


## Get Fine Solutions
#

print("Generating Fine Full Solution")
u_full_fine=full_system_solution(T,
                                 num_steps_time[-1],
                                 mesh_sizes[-1],
                                 mesh_sizes[-1],
                                 u_0,
                                 B,
                                 f,
                                 D)
print("Done Generating Fine Full Solution")
print("---")
print("Generating Fine Split Solution")
u_fos_fine=first_order_split_solution(T,
                                num_steps_time[-1],
                                mesh_sizes[-1],
                                mesh_sizes[-1],
                                u_0,
                                B,
                                f,
                                D)
print("Done Generating Fine Split Solution")

params=[]
current_iteration=1
for i in range(len(num_steps_time)):
    for j in range(len(mesh_sizes)):
        print("Doing iteration %s out of %s"%(current_iteration, len(num_steps_time)*len(mesh_sizes)))
        #print("Running with params: (%s, %s)"%(num_steps_time[i], mesh_sizes[j]))
        u_full= full_system_solution(T,
                                                                num_steps_time[i],
                                                                mesh_sizes[j],
                                                                mesh_sizes[j],
                                                                u_0,
                                                                B,
                                                                f,
                                                                D)
        l2_full_true=errornorm(u_true, u_full, 'L2')
        h1_full_true=errornorm(u_true, u_full, 'H1')
        l2_full_fine=errornorm(u_full_fine, u_full, 'L2')
        h1_full_fine=errornorm(u_full_fine, u_full, 'H1')
        
        print("Done with full")
        u_fos= first_order_split_solution(T,
                                                                    num_steps_time[i],
                                                                    mesh_sizes[j],
                                                                    mesh_sizes[j],
                                                                    u_0,
                                                                    B,
                                                                    f,
                                                                    D)
        l2_fos=errornorm(u_true, u_fos, 'L2')
        h1_fos=errornorm(u_true, u_fos, 'H1')
        l2_fos_fine=errornorm(u_fos_fine, u_fos, 'L2')
        h1_fos_fine=errornorm(u_fos_fine, u_fos, 'H1')
        
        print("Done with first order split")
        current_iteration+=1
        params.append([num_steps_time[i],
                       mesh_sizes[j],
                       l2_full_true,
                       h1_full_true,
                       l2_full_fine,
                       h1_full_fine,
                       l2_fos,
                       h1_fos,
                       l2_fos_fine,
                       h1_fos_fine])
        

params=np.array(params)

np.save('./data/sols_results_2d_%s'%B_text, params)
