# Convergence rates
# 
# In this notebook we numerically verify the theoretical converge rates of the finite element discretization of an elliptic problem.
# 
# More specifically, we consider the Poisson equation
# $$ - \frac{\partial^2 u}{\partial^2 x} = f(x)  \quad \forall x \in (0,1), $$
# with homegeneuos Dirichlet boundary conditions $u(0) = u(1) = 0$ and forcing term $f(x) = \sin(5 \pi x)$.
# The analytical solution is $u_{\rm ex} = \frac{1}{(5\pi)^2} \sin(5\pi x)$.
#
# 
# Assuming that the analytical solution is regular enough (i.e. $u_{\rm ex} \in H^{k+1}$), the following error estimates hold
# 
# - Energy norm:
# $$ \left\| u_{\rm ex} - u_h \right\|_{\rm energy} := \left( \int_0^1 \left|\frac{du_{\rm ex}}{dx} - \frac{du_{h}}{dx}\right|^2 dx \right)^{\frac{1}{2}} = \mathcal{O}(h^k), $$
# 
# - $L^2(\Omega)$ norm:
# $$ \left\| u_{\rm ex} - u_h \right\|_{L^2} := \left( \int_0^1 \left|u_{\rm ex} - u_{h}\right|^2 dx \right)^{\frac{1}{2}} = \mathcal{O}(h^{k+1}), $$
# 
# where $h$ denote the mesh size. For a uniform 1D mesh of the unit interval $h = \frac{1}{n}$ where $n$ is the number of mesh elements.

# Step 1. Import modules
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np


# Step 2. Finite element solution of the BVP
def solveBVP(n, degree):
    # 1. Mesh and finite element space
    mesh = UnitIntervalMesh(n)
    Vh = FunctionSpace(mesh, 'CG', degree)

    # 2. Define the right hand side f
    f = Expression("sin(omega*x[0])", omega = 5*pi, degree = 5)
    
    # 3. Define the Dirichlet boundary condition
    u_bc = Constant(0.)
    def dirichlet_boundary(x,on_boundary):
        return on_boundary
    bcs = [DirichletBC(Vh, u_bc, dirichlet_boundary)]

    # 4. Define the weak form
    uh = TrialFunction(Vh)
    vh = TestFunction(Vh)
    A_form = inner(grad(uh), grad(vh))*dx
    b_form = inner(f,vh)*dx

    # 5. Assemble and solve the finite element discrete problem
    A, b = assemble_system(A_form, b_form, bcs)
    uh = Function(Vh)
    solve(A, uh.vector(), b, "cg", "petsc_amg")

    # 6. Compute error norms
    u_ex = Expression("(1./(omega*omega))*sin(omega*x[0])", omega = 5*pi, degree = 5)
    err_L2 = np.sqrt( assemble((uh-u_ex)**2*dx) ) 

    grad_u_ex = Expression( "(1./(omega))*cos(omega*x[0])" , omega = 5*pi, degree = 5)
    err_energy = np.sqrt( assemble((grad(uh)[0]-grad_u_ex)**2*dx) )
    
    return err_L2, err_energy


# Step 3. Generate the converges plots
def make_convergence_plot(n, degree):
    errs_L2 = np.zeros(n.shape)
    errs_En = np.zeros(n.shape)
    
    for i in np.arange(n.shape[0]):
        print i, ": Solving problem on a mesh with", n[i], " elements."
        eL2, eE = solveBVP(n[i], degree)
        errs_L2[i] = eL2
        errs_En[i] = eE
        
    h = np.ones(n.shape)/n
    plt.loglog(h, errs_L2, '-*b', label='L2')
    plt.loglog(h, errs_En, '-*r', label='Energy')
    plt.loglog(h, .7*np.power(h,degree+1)*errs_L2[0]/np.power(h[0],degree+1), '--b', label = 'order {0}'.format(degree+1))
    plt.loglog(h, .7*np.power(h,degree)*errs_En[0]/np.power(h[0],degree), '--r', label = 'order {0}'.format(degree))
    plt.legend()
    plt.grid()
    plt.savefig("converge_rates_P{0}.png".format(degree))
    plt.close()
    


# Converges rate of piecewise linear finite element (k=1)
degree = 1
n = np.power(2, np.arange(2,7)) # n = [2^2, 2^3, 2^4 2^5, 2^6]
print "Generate converge plot for P1 Finite Elements"
make_convergence_plot(n, degree)


# Converges rate of piecewise quadratic finite element (k=2)
degree = 2
print "Generate converge plot for P2 Finite Elements"
make_convergence_plot(n, degree)

