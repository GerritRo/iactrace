Quick Start
===========

This guide walks through the basic IACTrace workflow: loading a telescope,
simulating observations, and visualizing results.

Loading a Telescope and Camera
------------------------------

Telescope and camera configurations live in separate YAML files. Load
each one independently — this lets you pair a single shared camera
(for example ``configs/CTAO/LSTCam.yaml``, used by all four LSTs)
with multiple telescopes.

.. code-block:: python

   import jax
   from iactrace import Telescope, Camera

   # Random key for Monte Carlo sampling.
   key = jax.random.key(0)

   # Load the telescope (optics + camera frame).
   telescope = Telescope.from_yaml(
       "configs/HESS/CT3.yaml", n_samples=256, key=key,
   )

   # Load the camera (sensor layout in the camera-local frame).
   camera = Camera.from_yaml("configs/HESS/HESS1U.yaml")

The ``n_samples`` parameter sets the number of rays traced per mirror facet.
Start with lower values (64-512) for quick iteration, increase for final
results.

Simulating Parallel Light
-------------------------

For astronomical sources, use parallel light with direction vectors:

.. code-block:: python

   import jax.numpy as jnp

   # On-axis source (light coming straight down)
   directions = jnp.array([[0.0, 0.0, -1.0]])
   values = jnp.array([1.0])

   # Trace through optics, then form a pixel image.
   ray_bundle = telescope.render(directions, values, source_type='parallel')
   image = camera.image(ray_bundle)

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

   ray_bundle = telescope.render(positions, values, source_type='point')
   image = camera.image(ray_bundle)

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

   ray_bundle = telescope.render(directions, intensities, source_type='parallel')
   image = camera.image(ray_bundle)

Visualizing Results
-------------------

IACTrace provides a unified :func:`~iactrace.viz.show_image` function
that renders both hexagonal IACT cameras and square pixel grids
(monitoring cameras, SiPM arrays):

.. code-block:: python

   import matplotlib.pyplot as plt
   from iactrace.viz import show_image

   fig, ax = plt.subplots(figsize=(8, 8))
   show_image(image, camera.sensor_groups[0], ax=ax, colorbar=True,
               cbar_label='Intensity')
   plt.show()

The pixel layout (hexagonal vs. square) is detected automatically from
the sensor group type.


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

The camera itself can be rendered the same way, with every pixel's light
concentrator and photosensor as described in the camera file:

.. code-block:: python

   from iactrace.viz import show_camera, show_sensor_chain

   show_camera(camera).show()          # the whole camera
   show_sensor_chain(camera).show()    # one pixel's chain, close up

Visualizing Ray Paths
---------------------

Both stages can report the path rays actually took, through
:meth:`~iactrace.Telescope.trace` for the optics and
:meth:`~iactrace.Camera.trace` for the camera:

.. code-block:: python

   from iactrace.viz import show_camera, show_telescope

   rays, traj = telescope.trace(origins, directions, values,
                                       record_trajectory=True)
   scene = show_telescope(telescope, trajectory=traj)
   scene.show()

   rays, traj = camera.trace(rays)
   show_camera(camera, trajectory=traj)
   scene.show()

Applying Optical Imperfections
------------------------------

Real telescopes have manufacturing tolerances and alignment errors:

.. code-block:: python

   # Surface roughness (broadens PSF), 12 arcsec on the primary
   telescope = telescope.apply_roughness(stage=0, sigma=12.0)

   # Mirror misalignment
   key = jax.random.key(42)
   telescope = telescope.apply_misalignment(
       stage=0,          # primary
       sigma_h=10.0,     # horizontal tip (arcsec)
       sigma_v=10.0,     # vertical tilt (arcsec)
       key=key,
   )

Multiple Sensor Groups
----------------------

Some cameras have more than one sensor group. The telescope renders rays
once; the camera selects which sensor group accumulates the image
via ``sensor_idx``:

.. code-block:: python

   ray_bundle = telescope.render(sources, values, source_type='parallel')

   image_sensor1 = camera.image(ray_bundle, sensor_idx=0)
   image_sensor2 = camera.image(ray_bundle, sensor_idx=1)

Next Steps
----------

- :doc:`telescope_operations` - Operations modifying telescopes
- :doc:`custom_telescopes` - Define your own telescope configurations
- :doc:`/examples/index` - Detailed examples for specific use cases
- :doc:`/api/index` - Full API reference