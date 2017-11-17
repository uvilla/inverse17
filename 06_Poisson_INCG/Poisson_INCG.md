---
title: Poisson INCG
layout: default
use_math: true
---

# Coefficient field inversion in an elliptic partial differential equation

We consider the estimation of a coefficient in an elliptic partial
differential equation as a model problem. Depending on the
interpretation of the unknowns and the type of measurements, this
model problem arises, for instance, in inversion for groundwater flow
or heat conductivity.  It can also be interpreted as finding a
membrane with a certain spatially varying stiffness. Let
$\Omega\subset\mathbb{R}^n$, $n\in\{1,2,3\}$ be an open, bounded
domain and consider the following problem:

$$
\min_{m} J(m):=\frac{1}{2}\int_\Omega (u-u_d)^2\, dx + \frac{\gamma}{2}\int_\Omega|\nabla m|^2\,dx,
$$

where $u$ is the solution of

$$
\begin{split}
\quad -\nabla\cdot(\exp(m)\nabla u) &= f \text{ in }\Omega,\\
u &= 0 \text{ on }\partial\Omega.
\end{split}
$$

Here $m\in U_{ad}:=\{m\in H^1(\Omega) \bigcap L^{\infty}(\Omega)\}$ the unknown coefficient field, $u_d$ denotes (possibly noisy) data, $f\in H^{-1}(\Omega)$ a given force, and $\gamma\ge 0$ the regularization parameter.

### The variational (or weak) form of the state equation:

Find $u\in H_0^1(\Omega)$ such that 

$$(\exp(m)\nabla u,\nabla v) - (f,v) = 0, \text{ for all } v\in H_0^1(\Omega),$$

where $H_0^1(\Omega)$ is the space of functions vanishing on $\partial\Omega$ with square integrable derivatives. Here, $(\cdot\,,\cdot)$ denotes the $L^2$-inner product, i.e, for scalar functions $u,v \in L^2(\Omega)$  we denote 

$$(u,v) := \int_\Omega u(x) v(x) \,dx.$$

### Gradient evaluation:

The Lagrangian functional $\mathscr{L}:H_0^1(\Omega)\times H^1(\Omega)\times H_0^1(\Omega)\rightarrow \mathbb{R}$ is given by

$$
\mathscr{L}(u,m,p):= \frac{1}{2}(u-u_d,u-u_d) +
\frac{\gamma}{2}(\nabla m, \nabla m) +  (\exp(m)\nabla u,\nabla p) - (f,p).
$$

Then the gradient of the cost functional $\mathcal{J}(m)$ with respect to the parameter $m$ is

$$
    \mathcal{G}(m)(\tilde m) := \gamma(\nabla m, \nabla \tilde{m}) +
     (\tilde{m}\exp(m)\nabla u, \nabla p) \quad \forall \tilde{m} \in H^1(\Omega),
$$

where $u \in H_0^1(\Omega)$ is the solution of the forward problem,

$$ \mathscr{L}_p(u,m,p)(\tilde{p})  := (\exp(m)\nabla u, \nabla \tilde{p}) - (f,\tilde{p}) = 0
\quad \forall \tilde{p} \in H_0^1(\Omega), $$

and $p \in H_0^1(\Omega)$ is the solution of the adjoint problem,

$$ \mathscr{L}_u(u,m,p)(\tilde{u}) := (\exp(m)\nabla p, \nabla \tilde{u}) + (u-u_d,\tilde{u}) = 0
\quad \forall \tilde{u} \in H_0^1(\Omega).$$

### Hessian action:

To evaluate the action $\mathcal{H}(m)(\hat{m})$ of the Hessian is a given direction $\hat{m}$ , we consider variations of the meta-Lagrangian functional

$$
\begin{aligned}
\mathscr{L}^H(u,m,p; \hat{u}, \hat{m}, \hat{p}) := & {} & {} \\
{} & \gamma(\nabla m, \nabla \tilde{m}) + (\tilde{m}\exp(m)\nabla u, \nabla p) & \text{gradient}\\
{} & + (\exp(m)\nabla u, \nabla \hat{p}) - (f,\hat{p}) & \text{forward eq}\\
{} & + (\exp(m)\nabla p, \nabla \hat{u}) + (u-u_d,\hat{u}) & \text{adjoint eq}.
\end{aligned}
$$

Then action of the Hessian is a given direction $\hat{m}$ is

$$
\begin{aligned}
(\tilde{m}, \mathcal{H}(m)(\hat{m}) ) & := \mathscr{L}^H_m(u,m,p; \hat{u}, \hat{m}, \hat{p})(\tilde{m}) \\
{} & =
(\tilde{m} \exp(m) \nabla \hat{u}, \nabla{p}) + \gamma (\nabla \hat{m}, \nabla \tilde{m}) + (\tilde{m} \hat{m} \exp(m)\nabla u, \nabla p) + (\tilde{m}\exp(m) \nabla u, \nabla \hat{p}) \quad \forall \tilde{m} \in H^1(\Omega),
\end{aligned}
$$

where 

- $u\in H^1_0(\Omega)$ and $p \in H^1_0(\Omega)$ are the solution of the forward and adjoint problem, respectively;

- $\hat{u} \in H^1_0(\Omega)$ is the solution of the incremental forward problem,

$$
\mathscr{L}^H_p(u,m,p; \hat{u}, \hat{m}, \hat{p})(\tilde{p}) := (\exp(m) \nabla \hat{u}, \nabla \tilde{p}) + (\hat{m} \exp(m) \nabla u, \nabla \tilde p) = 0 \quad \forall \tilde{p} \in H_0^1(\Omega);
$$


- and $\hat{p} \in H^1_0(\Omega)$ is the solution of the incremental adjoint problem,
$$
\mathscr{L}^H_u(u,m,p; \hat{u}, \hat{m}, \hat{p})(\tilde{u}) := (\hat{u}, \tilde{u}) + (\hat{m} \exp(m)\nabla p, \nabla \tilde{u}) + (\exp(m) \nabla \tilde u, \nabla \hat{p}) = 0 \quad \forall \tilde{u} \in H_0^1(\Omega).
$$

### Inexact Newton-CG:

Written in abstract form, the Newton Method computes an update direction $\hat{m}_k$ by solving the linear system 

$$
(\tilde{m}, \mathcal{H}(m_k)(\hat{m}_k) ) = -\mathcal{G}(m_k)(\tilde m) \quad \forall \tilde{m} \in H^1(\Omega),
$$

where the evaluation of the gradient $\mathcal{G}(m_k)$ involve the solution $u_k$ and $p_k$ of the forward and adjoint problem (respectively) for $m = m_k$.
Similarly, the Hessian action $\mathcal{H}(m_k)(\hat{m}_k)$ requires to additional solve the incremental forward and adjoint problems.

### Discrete Newton system:
$
\def\tu{\tilde u}
\def\tm{\tilde m}
\def\tp{\tilde p}
\def\hu{\hat u}
\def\hp{\hat p}
\def\hm{\hat m}
$

$
\def\bu{ {\bf u} }
\def\bm{ {\bf m} }
\def\bp{ {\bf p} }
\def\btu{ {\bf \tilde u} }
\def\btm{ {\bf \tilde m} }
\def\btp{ {\bf \tilde p} }
\def\bhu{ {\bf \hat u} }
\def\bhm{ {\bf \hat m} }
\def\bhp{ {\bf \hat p} }
\def\bg{ {\bf g} }
$

$
\def\bA{ {\bf A} }
\def\bC{ {\bf C} }
\def\bH{ {\bf H} }
\def\bR{ {\bf R} }
\def\bW{ {\bf W} }
$

Let us denote the vectors corresponding to the discretization of the functions $u_k, m_k, p_k$ by $\bu_k, \bm_k, \bp_k$ and of the functions $\hu_k, \hm_k, \hp_k$ by $\bhu_k, \bhm_k,\bhp_k$.

Then, the discretization of the above system is given by the following symmetric linear system:

$$
  \bH_k \, \bhm_k = -\bg_k.
$$

The gradient $\bg_k$ is computed using the following three steps

- Given $\bm_k$ we solve the forward problem

$$ \bA_k \bu_k = {\bf f}, $$

where $\bA_k \bu_k$ stems from the discretization $(\exp(m_k)\nabla u_k, \nabla \tilde{p})$, and ${\bf f}$ stands for the discretization of the right hand side $f$.

- Given $\bm_k$ and $\bu_k$ solve the adjoint problem

$$ \bA_k^T \bp_k = - \bW_{\scriptsize\mbox{uu}}\,(\bu_k-\bu_d) $$

where $\bA_k^T \bp_k$ stems from the discretization of $(\exp(m_k)\nabla \tilde{u}, \nabla p_k)$, $\bW_{\scriptsize\mbox{uu}}$ is the mass matrix corresponding to the $L^2$ inner product in the state space, and $\bu_d$ stems from the data.

- Define the gradient 

$$ \bg_k = \bR \bm_k + \bC_k^T \bp_k, $$

where $\bR$ is the matrix stemming from discretization of the regularization operator $\gamma ( \nabla \hat{m}, \nabla \tilde{m})$, and $\bC_k$ stems from discretization of the term $(\tilde{m}\exp(m_k)\nabla u_k, \nabla p_k)$.

Similarly the action of the Hessian $\bH_k \, \bhm_k$ in a direction $\bhm_k$ (by using the CG algorithm we only need the action of $\bH_k$ to solve the Newton step) is given by

- Solve the incremental forward problem

$$ \bA_k \bhu_k = -\bC_k \bm_k, $$

where $\bC_k \bm_k$ stems from discretization of $(\hat{m} \exp(m_k) \nabla u_k, \nabla \tilde p)$.

- Solve the incremental adjoint problem

$$ \bA_k^T \bhp_k = -(\bW_{\scriptsize\mbox{uu}} \bhu_k + \bW_{\scriptsize\mbox{um}}\,\bhm_k),$$

where $\bW_{\scriptsize\mbox{um}}\,\bhm_k$ stems for the discretization of $(\hat{m}_k \exp(m_k)\nabla p_k, \nabla \tilde{u})$.

- Define the Hessian action

$$
  \bH_k \, \bhm = \underbrace{(\bR + \bW_{\scriptsize\mbox{mm}})}_{\text{Hessian of the regularization}} \bhm +
    \underbrace{(\bC_k^{T}\bA_k^{-T} (\bW_{\scriptsize\mbox{uu}}
    \bA_k^{-1} \bC_k - \bW_{\scriptsize\mbox{um}}) -
    \bW_{\scriptsize\mbox{mu}} \bA_k^{-1}
    \bC_k)}_{\text{Hessian of the data misfit}}\;\bhm.
$$

### Goals:

By the end of this notebook, you should be able to:

- solve the forward and adjoint Poisson equations
- understand the inverse method framework
- visualise and understand the results
- modify the problem and code

### Mathematical tools used:

- Finite element method
- Derivation of gradiant and Hessian via the adjoint method
- inexact Newton-CG
- Armijo line search

### List of software used:

- <a href="http://fenicsproject.org/">FEniCS</a>, a parallel finite element element library for the discretization of partial differential equations
- <a href="http://www.mcs.anl.gov/petsc/">PETSc</a>, for scalable and efficient linear algebra operations and solvers
- <a href="http://matplotlib.org/">Matplotlib</a>, a python package used for plotting the results

## Set up

### Import dependencies


```python
from dolfin import *
import numpy as np

import sys
sys.path.append( "../hippylib" )
from hippylib import *

import logging

import matplotlib.pyplot as plt
%matplotlib inline
sys.path.append( "../hippylib/tutorial" )
import nb

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)
```

### Model set up:

As in the introduction, the first thing we need to do is set up the numerical model.  In this cell, we set the mesh, the finite element functions $u, p, g$ corresponding to state, adjoint and coefficient/gradient variables, and the corresponding test functions and the parameters for the optimization.


```python
# create mesh and define function spaces
nx = 64
ny = 64
mesh = UnitSquareMesh(nx, ny)
Vm = FunctionSpace(mesh, 'Lagrange', 1)
Vu = FunctionSpace(mesh, 'Lagrange', 2)

# The true and inverted parameter
mtrue = interpolate(Expression('log(2 + 7*(pow(pow(x[0] - 0.5,2) + pow(x[1] - 0.5,2),0.5) > 0.2))', degree=5),Vm)
m = interpolate(Expression("log(2.0)", degree=1),Vm)

# define function for state and adjoint
u = Function(Vu)
p = Function(Vu)

# define Trial and Test Functions
u_trial, p_trial, m_trial = TrialFunction(Vu), TrialFunction(Vu), TrialFunction(Vm)
u_test, p_test, m_test = TestFunction(Vu), TestFunction(Vu), TestFunction(Vm)

# initialize input functions
f = Constant("1.0")
u0 = Constant("0.0")

# plot
plt.figure(figsize=(15,5))
nb.plot(mesh,subplot_loc=121, mytitle="Mesh", show_axis='on')
nb.plot(mtrue,subplot_loc=122, mytitle="True parameter field")
plt.show()
```


![png](Poisson_INCG_files/Poisson_INCG_4_0.png)



```python
# set up dirichlet boundary conditions
def boundary(x,on_boundary):
    return on_boundary

bc_state = DirichletBC(Vu, u0, boundary)
bc_adj = DirichletBC(Vu, Constant(0.), boundary)
```

### Set up synthetic observations:

- Propose a coefficient field $m_{\rm true}$ shown above
- The weak form of the pde: 
    Find $u\in H_0^1(\Omega)$ such that $\underbrace{(\exp(m_{\rm true})\nabla u,\nabla v)}_{\; := \; a_{pde}} - \underbrace{(f,v)}_{\; := \;L_{pde}} = 0, \text{ for all } v\in H_0^1(\Omega)$.

- Perturb the solution: $u = u + \eta$, where $\eta \sim \mathcal{N}(0, \sigma)$


```python
# noise level
noise_level = 0.05

# weak form for setting up the synthetic observations
a_goal = inner(exp(mtrue) * nabla_grad(u_trial), nabla_grad(u_test)) * dx
L_goal = f * u_test * dx

# solve the forward/state problem to generate synthetic observations
goal_A, goal_b = assemble_system(a_goal, L_goal, bc_state)

utrue = Function(Vu)
solve(goal_A, utrue.vector(), goal_b)

ud = Function(Vu)
ud.assign(utrue)

# perturb state solution and create synthetic measurements ud
# ud = u + ||u||/SNR * random.normal
MAX = ud.vector().norm("linf")
noise = Vector()
goal_A.init_vector(noise,1)
noise.set_local( noise_level * MAX * np.random.normal(0, 1, len(ud.vector().array())) )
bc_adj.apply(noise)

ud.vector().axpy(1., noise)

# plot
nb.multi1_plot([utrue, ud], ["State solution with atrue", "Synthetic observations"])
plt.show()
```


![png](Poisson_INCG_files/Poisson_INCG_7_0.png)


### The cost function evaluation:

$$
J(m):=\underbrace{\frac{1}{2}\int_\Omega (u-u_d)^2\, dx}_{\text{misfit} } + \underbrace{\frac{\gamma}{2}\int_\Omega|\nabla m|^2\,dx}_{\text{reg}}
$$

In the code below, $\bW$ and $\bR$ are symmetric positive definite matrices that stem from finite element discretization of the misfit and regularization component of the cost functional, respectively.


```python
# regularization parameter
gamma = 1e-8

# weak for for setting up the misfit and regularization compoment of the cost
W_equ   = inner(u_trial, u_test) * dx
R_equ   = gamma * inner(nabla_grad(m_trial), nabla_grad(m_test)) * dx

W = assemble(W_equ)
R = assemble(R_equ)

# refine cost function
def cost(u, ud, m, W, R):
    diff = u.vector() - ud.vector()
    reg = 0.5 * m.vector().inner(R*m.vector() ) 
    misfit = 0.5 * diff.inner(W * diff)
    return [reg + misfit, misfit, reg]
```

### Setting up the state equations, right hand side for the adjoint and the necessary matrices:


```python
# weak form for setting up the state equation
a_state = inner(exp(m) * nabla_grad(u_trial), nabla_grad(u_test)) * dx
L_state = f * u_test * dx

# weak form for setting up the adjoint equation
a_adj = inner(exp(m) * nabla_grad(p_trial), nabla_grad(p_test)) * dx
L_adj = -inner(u - ud, p_test) * dx

# weak form for setting up matrices
Wum_equ = inner(exp(m) * m_trial * nabla_grad(p_test), nabla_grad(p)) * dx
C_equ   = inner(exp(m) * m_trial * nabla_grad(u), nabla_grad(u_test)) * dx
Wmm_equ = inner(exp(m) * m_trial * m_test *  nabla_grad(u),  nabla_grad(p)) * dx

M_equ   = inner(m_trial, m_test) * dx

# assemble matrix M
M = assemble(M_equ)
```

### Initial guess
We solve the state equation and compute the cost functional for the initial guess of the parameter ``a_ini``


```python
# solve state equation
state_A, state_b = assemble_system (a_state, L_state, bc_state)
solve (state_A, u.vector(), state_b)

# evaluate cost
[cost_old, misfit_old, reg_old] = cost(u, ud, m, W, R)

# plot
plt.figure(figsize=(15,5))
nb.plot(m,subplot_loc=121, mytitle="m_ini", vmin=mtrue.vector().min(), vmax=mtrue.vector().max())
nb.plot(u,subplot_loc=122, mytitle="u(m_ini)")
plt.show()
```


![png](Poisson_INCG_files/Poisson_INCG_13_0.png)


### The reduced Hessian apply to a vector $\bhm$:

Here we describe how to apply the reduced Hessian operator to a vector $\bhm$. For an opportune choice of the regularization, the reduced Hessian operator evaluated in a neighborhood of the solution is positive define, whereas far from the solution the reduced Hessian may be indefinite. On the constrary, the Gauss-Newton approximation of the Hessian is always positive defined.

For this reason, it is beneficial to perform a few initial Gauss-Newton steps (5 in this particular example) to accelerate the convergence of the inexact Newton-CG algorithm.

The Hessian apply reads:
$$
\begin{align}
\bhu &= -\bA^{-1} \bC \bhm\, & \text{linearized forward}\\
\bhp &= -\bA^{-T} (\bW_{\scriptsize\mbox{uu}} \bhu +
\bW_{\scriptsize\mbox{um}}\,\bhm) & \text{adjoint}\\
\bH \bhm &= (\bR + \bW_{\scriptsize\mbox{mm}})\bhm + \bC^T \bhp + \bW_{\scriptsize\mbox{mu}} \bhu.
\end{align}
$$

The Gauss-Newton Hessian apply is obtained by dropping the second derivatives operators $\bW_{\scriptsize\mbox{um}}\,\bhm$, $\bW_{\scriptsize\mbox{mm}}\bf \bhm$, and $\bW_{\scriptsize\mbox{mu}} \bhu$:
$$
\begin{align}
\bhu &= -\bA^{-1} \bC \bf \bhm\, & \text{linearized forward}\\
\bhp &= -\bA^{-T} \bW_{\scriptsize\mbox{uu}} \bhu & \text{adjoint}\\
\bH_{\rm GN} \bhm &= \bR \bhm + \bC^T \bhp.
\end{align}
$$




```python
# Class HessianOperator to perform Hessian apply to a vector
class HessianOperator():
    cgiter = 0
    def __init__(self, R, Wmm, C, A, adj_A, W, Wum, use_gaussnewton=False):
        self.R = R
        self.Wmm = Wmm
        self.C = C
        self.A = A
        self.adj_A = adj_A
        self.W = W
        self.Wum = Wum
        self.use_gaussnewton = use_gaussnewton
        
        # incremental state
        self.du = Vector()
        self.A.init_vector(self.du,0)
        
        #incremental adjoint
        self.dp = Vector()
        self.adj_A.init_vector(self.dp,0)
        
        # auxiliary vectors
        self.CT_dp = Vector()
        self.C.init_vector(self.CT_dp, 1)
        self.Wum_du = Vector()
        self.Wum.init_vector(self.Wum_du, 1)
        
    def init_vector(self, v, dim):
        self.R.init_vector(v,dim)

    # Hessian performed on v, output as generic vector y
    def mult(self, v, y):
        self.cgiter += 1
        y.zero()
        if self.use_gaussnewton:
            self.mult_GaussNewton(v,y)
        else:
            self.mult_Newton(v,y)
            
    # define (Gauss-Newton) Hessian apply H * v
    def mult_GaussNewton(self, v, y):
        
        #incremental forward
        rhs = -(self.C * v)
        bc_adj.apply(rhs)
        solve (self.A, self.du, rhs)
        
        #incremental adjoint
        rhs = - (self.W * self.du)
        bc_adj.apply(rhs)
        solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1, self.CT_dp)
        
    # define (Newton) Hessian apply H * v
    def mult_Newton(self, v, y):
        
        #incremental forward
        rhs = -(self.C * v)
        bc_adj.apply(rhs)
        solve (self.A, self.du, rhs)
        
        #incremental adjoint
        rhs = -(self.W * self.du) -  self.Wum * v
        bc_adj.apply(rhs)
        solve (self.adj_A, self.dp, rhs)
        
        #Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1., self.Wmm*v)
        
        #Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)
```

## The inexact Newton-CG optimization with Armijo line search:

We solve the constrained optimization problem using the inexact Newton-CG method with Armijo line search.

The stopping criterion is based on a relative reduction of the norm of the gradient (i.e. $\frac{\|g_{n}\|}{\|g_{0}\|} \leq \tau$).

First, we compute the gradient by solving the state and adjoint equation for the current parameter $m$, and then substituing the current state $u$, parameter $m$ and adjoint $p$ variables in the weak form expression of the gradient:
$$ (g, \tilde{m}) = \gamma(\nabla m, \nabla \tilde{m}) +(\tilde{m}\nabla u, \nabla p).$$

Then, we compute the Newton direction $\hat m$ by iteratively solving $\mathcal{H} {\hat m} = -g$.
The Newton system is solved inexactly by early termination of conjugate gradient iterations via Eisenstat–Walker (to prevent oversolving) and Steihaug  (to avoid negative curvature) criteria.

Finally, the Armijo line search uses backtracking to find $\alpha$ such that a sufficient reduction in the cost functional is achieved.
More specifically, we use backtracking to find $\alpha$ such that:
$$J( m + \alpha \delta m ) \leq J(m) + \alpha c_{\rm armijo} (\hat m,g). $$


```python
# define parameters for the optimization
tol = 1e-8
c = 1e-4
maxiter = 12
plot_on = False

# initialize iter counters
iter = 1
total_cg_iter = 0
converged = False

# initializations
g, m_delta = Vector(), Vector()
R.init_vector(m_delta,0)
R.init_vector(g,0)

m_prev = Function(Vm)

print "Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg"

while iter <  maxiter and not converged:

    # assemble matrix C
    C =  assemble(C_equ)

    # solve the adoint problem
    adjoint_A, adjoint_RHS = assemble_system(a_adj, L_adj, bc_adj)
    solve(adjoint_A, p.vector(), adjoint_RHS)

    # assemble W_ua and R
    Wum = assemble (Wum_equ)
    Wmm = assemble (Wmm_equ)

    # evaluate the  gradient
    CT_p = Vector()
    C.init_vector(CT_p,1)
    C.transpmult(p.vector(), CT_p)
    MG = CT_p + R * m.vector()
    solve(M, g, MG)

    # calculate the norm of the gradient
    grad2 = g.inner(MG)
    gradnorm = sqrt(grad2)

    # set the CG tolerance (use Eisenstat–Walker termination criterion)
    if iter == 1:
        gradnorm_ini = gradnorm
    tolcg = min(0.5, sqrt(gradnorm/gradnorm_ini))

    # define the Hessian apply operator (with preconditioner)
    Hess_Apply = HessianOperator(R, Wmm, C, state_A, adjoint_A, W, Wum, use_gaussnewton=(iter<6) )
    P = R + gamma * M
    Psolver = PETScKrylovSolver("cg", amg_method())
    Psolver.set_operator(P)
    
    solver = CGSolverSteihaug()
    solver.set_operator(Hess_Apply)
    solver.set_preconditioner(Psolver)
    solver.parameters["rel_tolerance"] = tolcg
    solver.parameters["zero_initial_guess"] = True
    solver.parameters["print_level"] = -1

    # solve the Newton system H a_delta = - MG
    solver.solve(m_delta, -MG)
    total_cg_iter += Hess_Apply.cgiter
    
    # linesearch
    alpha = 1
    descent = 0
    no_backtrack = 0
    m_prev.assign(m)
    while descent == 0 and no_backtrack < 10:
        m.vector().axpy(alpha, m_delta )

        # solve the state/forward problem
        state_A, state_b = assemble_system(a_state, L_state, bc_state)
        solve(state_A, u.vector(), state_b)

        # evaluate cost
        [cost_new, misfit_new, reg_new] = cost(u, ud, m, W, R)

        # check if Armijo conditions are satisfied
        if cost_new < cost_old + alpha * c * MG.inner(m_delta):
            cost_old = cost_new
            descent = 1
        else:
            no_backtrack += 1
            alpha *= 0.5
            m.assign(m_prev)  # reset a

    # calculate sqrt(-G * D)
    graddir = sqrt(- MG.inner(m_delta) )

    sp = ""
    print "%2d %2s %2d %3s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %5.2f %1s %5.3e" % \
        (iter, sp, Hess_Apply.cgiter, sp, cost_new, sp, misfit_new, sp, reg_new, sp, \
         graddir, sp, gradnorm, sp, alpha, sp, tolcg)

    if plot_on:
        nb.multi1_plot([m,u,p], ["m","u","p"], same_colorbar=False)
        plt.show()
    
    # check for convergence
    if gradnorm < tol and iter > 1:
        converged = True
        print "Newton's method converged in ",iter,"  iterations"
        print "Total number of CG iterations: ", total_cg_iter
        
    iter += 1
    
if not converged:
    print "Newton's method did not converge in ", maxiter, " iterations"
```

    Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg
     1     1     1.12766e-05   1.12765e-05   1.34003e-11   1.56545e-02   3.79438e-04    1.00   5.000e-01
     2     1     7.84203e-07   7.84166e-07   3.67855e-11   4.68307e-03   5.35012e-05    1.00   3.755e-01
     3     1     3.14953e-07   3.14904e-07   4.91552e-11   9.71779e-04   7.13995e-06    1.00   1.372e-01
     4     6     1.93046e-07   1.62308e-07   3.07374e-08   4.58168e-04   1.00948e-06    1.00   5.158e-02
     5     1     1.87452e-07   1.56699e-07   3.07529e-08   1.05781e-04   6.27162e-07    1.00   4.066e-02
     6    12     1.81588e-07   1.38650e-07   4.29376e-08   1.09840e-04   2.15958e-07    1.00   2.386e-02
     7     5     1.81529e-07   1.39701e-07   4.18281e-08   1.08075e-05   3.73488e-08    1.00   9.921e-03
     8    17     1.81528e-07   1.39790e-07   4.17376e-08   1.45824e-06   2.88310e-09    1.00   2.757e-03
    Newton's method converged in  8   iterations
    Total number of CG iterations:  44



```python
nb.multi1_plot([mtrue, m], ["mtrue", "m"])
nb.multi1_plot([u,p], ["u","p"], same_colorbar=False)
plt.show()
```


![png](Poisson_INCG_files/Poisson_INCG_18_0.png)



![png](Poisson_INCG_files/Poisson_INCG_18_1.png)


Copyright (c) 2016, The University of Texas at Austin & University of California, Merced.
All Rights reserved.
See file COPYRIGHT for details.

This file is part of the hIPPYlib library. For more information and source code
availability see https://hippylib.github.io.

hIPPYlib is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (as published by the Free Software Foundation) version 2.0 dated June 1991.
