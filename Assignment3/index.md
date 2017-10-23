---
title: Home
layout: default
use_math: false
---

# Assignment #3 (Due November 6)
You will need the following files to solve Problems 2 and 3 using FEniCS:

- [tntv.ipynb](tntv.ipynb): This notebook includes some starter lines of python code for Problem 3 to define the mesh and finite element space and to evaluate the true and noisy images at each point of the mesh.

- [unconstrainedMinimization.py](unconstrainedMinimization.py): This file includes an implementation of inexact Newton-CG to solve variational unconstrained minimization problems using the Eisenstat-Walker termination condition and an Armijo-based line search (early termination due to negative curvature is not necessary, since Problem 3 results in positive definite Hessians).

- [energyMinimization.ipynb](energyMinimization.ipynb): This file includes an example of how to use the inexact Newton-CG nonlinear solver implemented in unconstrainedMinimization.py.

- [image.dat](image.dat) This is the target image for Problem 3 that is read into tntv.ipynb.
    
