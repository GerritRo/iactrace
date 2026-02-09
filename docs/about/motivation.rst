Motivation
==========

Why Differentiable Ray Tracing?
-------------------------------

IACTrace allows differentiable raytracing and source rendering by being based on  
`JAX <https://github.com/jax-ml/jax>`_. This has a variety of advantages:

**Gradient-based optimization**
   Rather than exhaustive grid searches or stochastic sampling, we can use
   efficient gradient descent to find optimal telescope configurations. This is
   particularly powerful for high-dimensional problems like aligning hundreds
   of individual mirror facets.

**Inverse problems**
   Given an observed image (e.g., from a calibration LED or star), we can infer
   the telescope state that produced it. This enables automated alignment
   monitoring and correction.

**Sensitivity analysis**
   Gradients directly quantify how sensitive the optical performance is to each
   parameter, informing manufacturing tolerances and maintenance priorities.

Why JAX?
--------

IACTrace is built on `JAX <https://github.com/jax-ml/jax>`_, a numerical
computing library that provides:

**Automatic differentiation**
   JAX can differentiate through arbitrary Python/NumPy code, including loops,
   conditionals, and custom data structures. This eliminates the need to
   manually derive and implement gradient formulas.

**Hardware acceleration**
   The same code runs on CPU, GPU, or TPU with minimal modification. This enables
   efficient parallel simulations of skyfields, reducing the time requirement for 
   simulating many sources at once.
   
Why Another Package for IACT Raytracing?
----------------------------------------

The reasons will be given more in-depth below, but on a personal note, 
I was unsatisfied with the current offer: Commercial raytracers were too expensive,
existing alternatives such as ROBAST / sim_telarray do not offer GPU support or differentiability.

**This package is meant to fill the gap, for people who:**

- Do not want to use commercial software
- Do not need electronics / airshower simulations
- Don't need very high accuracy / non-sequential raytracing

**But care about:**

- Speed via GPU-acceleration
- Differentiability
- Simplicity and good python integration.