Quick Start
===========

This guide walks through the basic IACTrace workflow: loading a telescope,
simulating observations, and visualizing results.

Loading a Telescope
-------------------

The simplest way to get started is loading a pre-configured telescope:

.. code-block:: python

   import jax
   from iactrace import MCIntegrator, load_telescope

   # Random key for Monte Carlo sampling
   key = jax.random.key(0)

   # Integrator controls ray sampling density
   integrator = MCIntegrator(n_samples=256)

   # Load telescope from YAML configuration
   telescope = load_telescope("configs/HESS/CT3.yaml", integrator, key)

The ``n_samples`` parameter sets the number of rays traced per mirror facet.
Start with lower values (64-512) for quick iteration, increase for final
results.

Simulating Parallel Light
-------------------------

For astronomical sources (stars, gamma-ray showers), use parallel light with
direction vectors:

.. code-block:: python

   import jax.numpy as jnp

   # On-axis source (light coming straight down)
   directions = jnp.array([[0.0, 0.0, -1.0]])
   values = jnp.array([1.0])

   # Render image
   image = telescope.render(directions, values, source_type='parallel')

For off-axis sources, compute the direction vector from the angular offset:

.. code-block:: python

   # Source at 2 degrees off-axis in X
   angle_deg = 2.0
   angle_rad = angle_deg * jnp.pi / 180

   direction = jnp.array([[jnp.sin(angle_rad), 0.0, -jnp.cos(angle_rad)]])

Simulating Point Sources
------------------------

For finite-distance calibration sources, use point source mode with positions:

.. code-block:: python

   # Point source at 500m distance, slightly off-axis
   positions = jnp.array([[-0.5, 0.3, 500.0]])
   values = jnp.array([1.0])

   image = telescope.render(positions, values, source_type='point')

Multiple Sources
----------------

Render multiple sources simultaneously by stacking them:

.. code-block:: python

   # Random star field
   n_stars = 100
   key1, key2 = jax.random.split(key)

   fov_rad = 3.0 * jnp.pi / 180  # 3 degree field of view

   x = jax.random.uniform(key1, (n_stars,), minval=-fov_rad/2, maxval=fov_rad/2)
   y = jax.random.uniform(key2, (n_stars,), minval=-fov_rad/2, maxval=fov_rad/2)
   z = -jnp.ones(n_stars)

   directions = jnp.stack([x, y, z], axis=1)
   directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

   # Random intensities
   intensities = jax.random.uniform(key, (n_stars,))

   image = telescope.render(directions, intensities, source_type='parallel')

Visualizing Results
-------------------

IACTrace provides visualization functions for both sensor types:

**Hexagonal sensors** (typical IACT cameras):

.. code-block:: python

   import matplotlib.pyplot as plt
   from iactrace import hexshow

   fig, ax = plt.subplots(figsize=(8, 8))
   hexshow(image, telescope.sensors[0], ax=ax)
   plt.colorbar(ax.collections[0], label='Intensity')
   plt.show()

**Square sensors** (monitoring cameras, SiPM arrays):

.. code-block:: python

   from iactrace import squareshow

   fig, ax = plt.subplots(figsize=(8, 8))
   squareshow(image, telescope.sensors[0], ax=ax)
   plt.colorbar(ax.images[0], label='Intensity')
   plt.show()

3D Telescope Visualization
--------------------------

Inspect the telescope geometry in 3D:

.. code-block:: python

   from iactrace.viz import show_telescope

   scene = show_telescope(telescope)
   scene.show()  # Opens interactive viewer

In Jupyter notebooks, use:

.. code-block:: python

   scene.show(viewer='jupyter')

Applying Optical Imperfections
------------------------------

Real telescopes have manufacturing tolerances and alignment errors:

.. code-block:: python

   # Surface roughness (broadens PSF)
   telescope = telescope.apply_roughness(24.0)  # 24 arcseconds RMS

   # Mirror misalignment
   key = jax.random.key(42)
   telescope = telescope.apply_misalignment_to_group(
       group_idx=0,      # Primary mirror group
       sigma_h=10.0,     # Horizontal misalignment (arcsec)
       sigma_v=10.0,     # Vertical misalignment (arcsec)
       key=key
   )

Multiple Sensors
----------------

Some telescopes have multiple sensors (science camera + lid camera). Select
which sensor to render:

.. code-block:: python

   # Render to science camera (index 0)
   image_science = telescope.render(sources, values, source_type='parallel', sensor_idx=0)

   # Render to monitoring camera (index 1)
   image_monitor = telescope.render(sources, values, source_type='parallel', sensor_idx=1)

Next Steps
----------

- :doc:`telescope_operations` - Operations modifying telescopes
- :doc:`custom_telescopes` - Define your own telescope configurations
- :doc:`/examples/index` - Detailed examples for specific use cases
- :doc:`/api/index` - Full API reference