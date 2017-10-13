# Poisson equation in 1D
# 
# In this tutorial, we show how to solve the Poisson equation in 1D
# 
# $$ - \frac{\partial^2 u}{\partial^2 x} = f(x)  \in (0,1), $$
# 
# using different types of boundary conditions.

# ## Homogeneus Dirichlet boundary conditions
# 
# Consider
# 
# $$ - \frac{\partial^2 u}{\partial^2 x} = f(x)  \in (0,1), $$
# with homegeneuos Dirichlet boundary conditions $u(0) = u(1) = 0$.
#  
# Below we solve the problem using FEniCS for the choice $f(x) = \sin(\pi x)$.
# The analytical solution is $u_{ex} = \frac{1}{\pi^2} \sin(\pi x)$.
# 

# 1. Imports
from dolfin import *
import matplotlib.pyplot as plt
import math

# 2. Define the mesh and the finite element space
n = 20
mesh = UnitIntervalMesh(n)
Vh = FunctionSpace(mesh, 'CG', 1)
print "Number of dofs", Vh.dim()


# 3. Define the right hand side f
f = Expression("sin(omega*x[0])", omega = pi, degree = 5)

# 4. Define the Dirichlet boundary condition
u_bc = Constant(0.)
def dirichlet_boundary(x,on_boundary):
    return on_boundary
bcs = [DirichletBC(Vh, u_bc, dirichlet_boundary)]

# 5. Define the weak form
uh = TrialFunction(Vh)
vh = TestFunction(Vh)
A_form = inner(grad(uh), grad(vh))*dx
b_form = inner(f,vh)*dx


# 6. Assemble and solve the finite element discrete problem
A, b = assemble_system(A_form, b_form, bcs)
uh = Function(Vh)
solve(A, uh.vector(), b)
File("solution_1.pvd") << uh


# 7. Compute error norms
u_ex = Expression("(1./(omega*omega))*sin(omega*x[0])", omega = pi, degree = 5)
err_L2 = math.sqrt( assemble((uh-u_ex)**2*dx) ) 
print "|| u_h - u_ex ||_L^2 = {0:5e}".format(err_L2)

grad_u_ex = Expression( "(1./(omega))*cos(omega*x[0])" , omega = pi, degree = 5)
err_energy = math.sqrt( assemble((grad(uh)[0]-grad_u_ex)**2*dx) )
print "|| u_h - u_ex ||_e = {0:5e}".format(err_energy)


# Mixed Dirichlet and Neumann boundary conditions
# 
# Consider
# 
# $$ - \frac{\partial^2 u}{\partial^2 x} = f(x)  \in (0,1), $$
# with boundary conditions:
# $$
# u(0) = 0, \quad \left.\frac{\partial u}{\partial x}\right|_{x=1} = g
# $$
#  
# Below we solve the problem using FEniCS for the choice $f(x) = 2$, $g = -1$.
# The analytical solution is $u_{ex} = x - x^2$


#1. Imports
from dolfin import *
import matplotlib.pyplot as plt
import math

#2. Define the mesh and the finite element space
n = 20
mesh = UnitIntervalMesh(n)
Vh = FunctionSpace(mesh, 'CG', 1)
print "Number of dofs", Vh.dim()

#3. Define the right hand side f and the non-homogeneus Neumann condition g
f = Constant(2.)
g = Constant(-1.)

#4. Define the Dirichlet boundary condition
u_bc = Constant(0.)
def dirichlet_boundary(x,on_boundary):
    return on_boundary and near(x[0], 0)
bcs = [DirichletBC(Vh, u_bc, dirichlet_boundary)]

#5. Define the weak form
uh = TrialFunction(Vh)
vh = TestFunction(Vh)

A_form = inner(grad(uh), grad(vh))*dx
b_form = inner(Constant(-1.), vh)*ds + inner(f,vh)*dx  

#6. Assemble and solve the finite element discrete problem
A, b = assemble_system(A_form, b_form, bcs)
uh = Function(Vh)
solve(A, uh.vector(), b)
File("solution_2.pvd") << uh

#7. Compute error norms
u_ex = Expression("x[0] - x[0]*x[0]", degree = 2)
err_L2 = math.sqrt( assemble((uh-u_ex)**2*dx) ) 
print "|| u_h - u_ex ||_L^2 = {0:5e}".format(err_L2)

grad_u_ex = Expression("1. - 2.*x[0]", degree = 1)
err_energy =math.sqrt( assemble((grad(uh)[0]-grad_u_ex)**2*dx) )
print "|| u_h - u_ex ||_e = {0:5e}".format(err_energy)