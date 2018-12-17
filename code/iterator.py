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


def strang_split_sol(T, dt, num_steps, nx, ny):
    set_log_level(50)

    # Start and End Points of RectangleMesh
    point_start=Point((0,0))
    point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))

    mesh = RectangleMesh(point_start, point_end, nx, ny)
    pbc=PeriodicBoundary()
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)

    # Define boundary condition
    u_0= Expression('sin(x[0])+cos(x[1])', degree=4)

    # True solution
    u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                      t=0, degree=4)

    # Define initial value
    u_n = project(u_0, V)
    u = TrialFunction(V)
    v = TestFunction(V)


    B_1=1
    B_2=1
    B=Constant((B_1, B_2))
    D=Constant(1.0)

    D_sys1=Constant(1.0)
    f=Expression('exp(-t)*(B_1*cos(x[0])-B_2*sin(x[1]))',
                 B_1=Constant(B_1), B_2=Constant(B_2), t=0, degree=4)

    A_a=u*v*dx+D*dot(grad(u), grad(v))*dt*dx
    A_L=u_n*v*dx
    A_a_ass=assemble(A_a)
    
    B1_a=u*v*dx+dot(B, grad(u))*v*dt/float(2)*dx
    B1_L=u_n*v*dx+f*v*dt/float(2)*dx
    B1_a_ass=assemble(B1_a)

    B2_a=u*v*dx+dot(B, grad(u))*v*dt/float(2)*dx
    B2_L=u_n*v*dx
    B2_a_ass=assemble(B2_a)

    u=Function(V)
    t=0

    for n in range(num_steps):

    
        # Do system 2 update, part 1
        f.t=t
        b_temp=assemble(B1_L) 
        solve(B1_a_ass, u.vector(), b_temp)
        u_n.assign(u)

        
        # Do system 1 update
        b_temp=assemble(A_L)
        solve(A_a_ass, u.vector(), b_temp)
        u_n.assign(u)

        # Do system 2 update
        b_temp=assemble(B2_L)
        solve(B2_a_ass, u.vector(), b_temp)
        u_n.assign(u)

        t+=dt

        u_true.t=t

      
    error_l2=errornorm(u_true, u_n, 'L2')
    error_h1=errornorm(u_true, u_n, 'H1')
    vertex_values_u_true=u_true.compute_vertex_values(mesh)
    vertex_values_u_n=u_n.compute_vertex_values(mesh)
    error_max=np.max(np.abs(vertex_values_u_true-vertex_values_u_n))
    return error_l2, error_h1, error_max

def full_system_solution(T, dt,num_steps, nx, ny):

    set_log_level(50)
    # Start and End Points of RectangleMesh
    point_start=Point((0,0))
    point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))

    mesh = RectangleMesh(point_start, point_end, nx, ny)
    pbc=PeriodicBoundary()
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)

    # Define boundary condition
    u_0= Expression('sin(x[0])+cos(x[1])', degree=4)

    # True solution
    u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                      t=0, degree=4)

    # Define initial value
    u_n = project(u_0, V)
    u = TrialFunction(V)
    v = TestFunction(V)

    B_1=1
    B_2=1
    B=Constant((B_1, B_2))
    D=Constant(1.0)

    D_sys1=Constant(1.0)
    f=Expression('exp(-t)*(B_1*cos(x[0])-B_2*sin(x[1]))',
                 B_1=Constant(B_1), B_2=Constant(B_2), t=0, degree=4)

    a=D*u*v*dx+dt*dot(grad(u), grad(v))*dx+dot(B, grad(u))*v*dt*dx
    L=u_n*v*dx+f*v*dt*dx
    A_sys=assemble(a)

    u=Function(V)
    t=0

    for n in range(num_steps):
        t+=dt
        f.t=t
        solve(a==L,u)
        u_n.assign(u)
        u_true.t=t


    error_l2=errornorm(u_true, u_n, 'L2')
    error_h1=errornorm(u_true, u_n, 'H1')
    vertex_values_u_true=u_true.compute_vertex_values(mesh)
    vertex_values_u_n=u_n.compute_vertex_values(mesh)
    error_max=np.max(np.abs(vertex_values_u_true-vertex_values_u_n))
    return error_l2, error_h1, error_max
    
def first_order_split_solution(T, dt,num_steps, nx, ny):
    set_log_level(50)

    # Start and End Points of RectangleMesh
    point_start=Point((0,0))
    point_end=Point((2*DOLFIN_PI, 2*DOLFIN_PI))

    mesh = RectangleMesh(point_start, point_end, nx, ny)
    pbc=PeriodicBoundary()
    V = FunctionSpace(mesh, 'P', 1, constrained_domain=pbc)

    # Define boundary condition
    u_0 = Expression('sin(x[0])+cos(x[1])',
                     degree=4)

    # True solution
    u_true=Expression('exp(-t)*(sin(x[0])+cos(x[1]))',
                      t=0, degree=4)

    # Define initial value
    u_n = project(u_0, V)
    u = TrialFunction(V)
    v = TestFunction(V)


    B_1=1
    B_2=1
    B=Constant((B_1, B_2))
    D=Constant(1.0)

    D_sys1=Constant(1.0)


    f=Expression('exp(-t)*(B_1*cos(x[0])-B_2*sin(x[1]))',
                 B_1=Constant(B_1), B_2=Constant(B_2), t=0, degree=4)


    a_A=u*v*dx+dt*D_sys1*dot(grad(u), grad(v))*dx
    L_A=u_n*v*dx+dt*f*v*dx
    a_A_ass=assemble(a_A)

    a_B=u*v*dx+dt*dot(B, grad(u))*v*dx
    L_B=u_n*v*dx
    a_B_ass=assemble(a_B)

    u=Function(V)
    t=0

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

    error_l2=errornorm(u_true, u_n, 'L2')
    error_h1=errornorm(u_true, u_n, 'H1')
    vertex_values_u_true=u_true.compute_vertex_values(mesh)
    vertex_values_u_n=u_n.compute_vertex_values(mesh)
    error_max=np.max(np.abs(vertex_values_u_true-vertex_values_u_n))
    return error_l2, error_h1, error_max


T=10
DT=[1,1/float(5), 1/float(10),1/float(50), 1/float(100),1/float(500), 1/float(1000)]
num_steps=[int(round(x)) for x in np.divide([T]*len(DT), np.array(DT))]
nx=ny=64


h1_full=[]
l2_full=[]
max_vertex_full=[]

h1_first_order=[]
l2_first_order=[]
max_vertex_first_order=[]

h1_strang=[]
l2_strang=[]
max_vertex_strang=[]

#def full_system_solution(T, dt, nx, ny):
for i in range(len(num_steps)):
    print("Doing iteration %s out of %s"%(i, len(num_steps)))
    print(num_steps[i])
    temp_l2, temp_h1, temp_vertex_max= full_system_solution(T, DT[i], num_steps[i], nx, ny)
    print("Done with full")
    h1_full.append(temp_h1)
    l2_full.append(temp_l2)
    max_vertex_full.append(temp_vertex_max)
    temp_l2, temp_h1, temp_vertex_max = first_order_split_solution(T, DT[i], num_steps[i],nx,ny)
    print("Done with first order split")
    h1_first_order.append(temp_h1)
    l2_first_order.append(temp_l2)
    max_vertex_first_order.append(temp_vertex_max)

    temp_l2, temp_h1, temp_vertex_max = strang_split_sol(T, DT[i], num_steps[i], nx, ny)

    print("Done with strang split")

    h1_strang.append(temp_h1)
    l2_strang.append(temp_l2)
    max_vertex_strang.append(temp_vertex_max)
    
    
print("Results for H1 - Full System")
print(h1_full)
print("-----")
print("Results for L2 - Full System")
print(l2_full)
print("-----")
print("Results for Max Vertex")
print(max_vertex_full)
print("----")
print("With DT")
print(DT)

print("\n")
print("------")


print("Results for H1 - First Order System")
print(h1_first_order)
print("-----")
print("Results for L2 - First Order System")
print(l2_first_order)
print("-----")
print("Results for Max Vertex - First Order System")
print(max_vertex_first_order)
print("----")
print("With DT")
print(DT)

print("\n")
print("-----")


print("Results for H1 - Strang System")
print(h1_strang)
print("-----")
print("Results for L2 - Strang System")
print(l2_strang)
print("-----")
print("Results for Max Vertex - Strang System")
print(max_vertex_strang)
print("----")
print("With DT")
print(DT)
