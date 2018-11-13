# Plan for the PDE project 


-Look at theory of semigroups more closely. Also at Strang splitting and its underpinnings.

-Fenics/python for periodic BC's and other more general syntax

- Look at the problem with periodic BC's to avoid problems

- investigate if problems cancel out if we don't have periodic BC's

- recreate the convergence that should be found analytically

- b should be variable since otherwise the commutativity of A and B is exact

- so solve the original equation and then do the splitting and compare with the exact result

- if we solve with only Initial data on inflow boundary will the solution of the other part correct if we use the first solution as start value for the second

- show that the convergence for hyperbolic equations with FEM is O(h) at best.

- 

We want to take the paper by Thomm√ and generalise for FEM 

Then we want to implement in fenics. 
First the heat equation (homogeneous)
-Which geometry? - (0,2pi)^d
For the equation in the paper, with periodic BC's?
- which periodic functions would be interesting


