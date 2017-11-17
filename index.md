---
title: Home
layout: default
use_math: false
---

This page collects the *Jupyter notebook* used for the graduate course on [**Computational and Variational Methods for Inverse Problems**](https://piazza.com/utexas/fall2017/geo391cse397me397ori397/home), taught by Prof. Ghattas at UT Austin in the Fall 2017 semester.

### Notebooks

- [Inverse problem prototype](01_InverseProblemPrototype/inverseProblemPrototype.html): An illustrative example of an ill-posed inverse problem ([.ipynb](01_InverseProblemPrototype/inverseProblemPrototype.ipynb)).

- Introduction to FEniCS:
	- [Poisson1D](02_IntroToFenics/Poisson1D.html): Finite element solution of the Poisson equation in 1D ([.ipynb](02_IntroToFenics/Poisson1D.ipynb)).
	- [Convergence Rates](02_IntroToFenics/ConvergenceRates.html): Convergence rates of the finite element method for the Poisson equation in 1D ([.ipynb](02_IntroToFenics/ConvergenceRates.ipynb)).
	- [Poisson2D](02_IntroToFenics/Poisson2D.html): Finite element solution of the Poisson equation in 2D ([.ipynb](02_IntroToFenics/Poisson2D.ipynb)).


- [Spectrum of Hessian operator](03_HessianSpectrum/HessianSpectrum.html): This notebook illustrates the spectral properties of the preconditioned Hessian misfit operator ([.ipynb](03_HessianSpectrum/HessianSpectrum.ipynb)). This notebook requires [hIPPYlib](https://hippylib.github.io).

- [Unconstrained Minimization](04_UnconstrainedMinimization/UnconstrainedMinimization.html): This notebook illustrates the  minimization of a non-quadratic energy functional using Netwon Method ([.ipynb](04_UnconstrainedMinimization/UnconstrainedMinimization.ipynb)).

- [Poisson SD](05_Poisson_SD/Poisson_SD.html): This notebook illustrates the use of FEniCS for solving an inverse problem for the coefficient field of a Poisson equation, using the steepest descent method  ([.ipnb](05_Poisson_SD/Poisson_SD.ipynb)). *Note that SD is a poor choice of optimization method for this problem*; it is provided here in order to compare with Newton's method, which we'll be using later in the class.

- [Poisson INCG](06_Poisson_INCG/Poisson_INCG.html): This notebook illustrates the use of FEniCS for solving an inverse problem for the coefficient field of a Poisson equation, using the inexact Newton CG method  ([.ipnb](06_Poisson_INCG/Poisson_INCG.ipynb)). This notebook requires [hIPPYlib](https://hippylib.github.io).

### Instructions

See [here](fenics_getting_started.pdf) for a list of introductory material to FEniCS and installation guidelines.

See [here](https://jupyter.readthedocs.io/en/latest/running.html#running) for instructions on how to use jupyther notebooks (files *.ipynb).

