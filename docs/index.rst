IACTrace
========

**Differentiable optical ray tracing for Imaging Atmospheric Cherenkov Telescopes**

IACTrace is a ray tracing library for simulating the optical
properties of IACT systems. Built on `JAX <https://jax.readthedocs.io/>`_ and
`Equinox <https://docs.kidger.site/equinox/>`_, it enables gradient-based
optimization and differentiable simulations.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from iactrace import Telescope, Camera

   # Load telescope and camera.
   key = jax.random.key(0)
   telescope = Telescope.from_yaml(
       "configs/HESS/CT3.yaml", n_samples=256, key=key,
   )
   camera = Camera.from_yaml("configs/HESS/HESS1U.yaml")

   # Simulate an on-axis parallel source.
   sources = jnp.array([[0.0, 0.0, -1.0]])  # direction vector
   values = jnp.array([1.0])                # intensity

   # Trace through optics, then form a pixel image.
   ray_bundle = telescope.render(sources, values, source_type="parallel")
   image = camera.image(ray_bundle)

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
   getting_started/conventions
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