# Poisson Equation in 2D
# 
# In this tutorial we solve the Poisson equation in two space dimensions.
# 
# For a domain $\Omega \subset \mathbb{R}^2$ with boundary $\partial \Omega = \Gamma_D \cup \Gamma_N$, we write the boundary value problem (BVP):
# 
# $$ 
# \left\{
# \begin{array}{ll}
# - \Delta u = f  & \text{in} \; \Omega, \\
# u = u_D & \text{on} \; \Gamma_D, \\
# \nabla u \cdot \boldsymbol{n} = g & \text{on} \; \Gamma_N.
# \end{array}
# \right.$$
#

# 1. Imports
from dolfin import *
import math


# 2. Define the mesh and the finite element space
n = 32
d = 1
mesh = UnitSquareMesh(n, n)
Vh = FunctionSpace(mesh, "Lagrange", d)
print "Number of dofs", Vh.dim()
plot(mesh)


# 3. Define the Dirichlet boundary condition
def boundary_d(x, on_boundary):
    return (x[1] < DOLFIN_EPS or x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS) and on_boundary

u_d  = Expression("sin(DOLFIN_PI*x[0])", degree = d+2)
bcs = [DirichletBC(Vh, u_d, boundary_d)]


# 4. Define the variational problem
uh = TrialFunction(Vh)
vh = TestFunction(Vh)
f = Constant(0.)
g = Expression("DOLFIN_PI*exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])", degree=d+2)
a = inner(grad(uh), grad(vh))*dx
L = f*vh*dx + g*vh*ds


# 5. Assemble and solve the finite element discrete problem
A, b = assemble_system(a, L, bcs)
uh = Function(Vh, name="solution")
solve(A, uh.vector(), b)

File("solution.pvd") << uh


# 6. Compute error norms
u_ex = Expression("exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])", degree = d+2, domain=mesh)
grad_u_ex = Expression( ("DOLFIN_PI*exp(DOLFIN_PI*x[1])*cos(DOLFIN_PI*x[0])",
                         "DOLFIN_PI*exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])"), degree = d+2, domain=mesh )

norm_u_ex   = math.sqrt(assemble(u_ex**2*dx))
norm_grad_ex = math.sqrt(assemble(inner(grad_u_ex, grad_u_ex)*dx))

err_L2   = math.sqrt(assemble((uh - u_ex)**2*dx))
err_grad = math.sqrt(assemble(inner(grad(uh) - grad_u_ex, grad(uh) - grad_u_ex)*dx))

print "|| u_ex - u_h ||_L2 / || u_ex ||_L2 = ", err_L2/norm_u_ex
print "|| grad(u_ex - u_h)||_L2 / = || grad(u_ex)||_L2 ", err_grad/norm_grad_ex