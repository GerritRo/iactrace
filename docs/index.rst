IACTrace
========

**Differentiable optical ray tracing for Imaging Atmospheric Cherenkov Telescopes**

IACTrace is a ray tracing library for simulating the optical
properties of IACT systems. Built on `JAX <https://jax.readthedocs.io/>`_ and
`Equinox <https://docs.kidger.site/equinox/>`_, it enables gradient-based
optimization and differentiable simulations.

.. code-block:: python

   import jax
   from iactrace import MCIntegrator, load_telescope

   # Load a telescope configuration
   key = jax.random.key(0)
   integrator = MCIntegrator(n_samples=1024)
   telescope = load_telescope("configs/HESS/CT3.yaml", integrator, key)

   # Simulate a star field
   sources = jax.numpy.array([[0.0, 0.0, -1.0]])  # Direction vector
   values = jax.numpy.array([1.0])                 # Intensity

   # Render the image
   image = telescope.render(sources, values, source_type='parallel')

----

.. toctree::
   :maxdepth: 2
   :caption: About This Project

   about/introduction
   about/motivation
   about/alternatives
   about/features

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/concept
   getting_started/quickstart
   getting_started/custom_telescopes
   getting_started/telescope_operations

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`