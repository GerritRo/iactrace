Telescope Operations
====================

IACTrace provides a functional API for modifying telescopes. All
operations return **new telescope instances** rather than mutating in
place, enabling reproducible simulations and compatibility with JAX
transformations.

Operations are addressed by **stage** — the integer ``optical_stage``
of the target group. The split between mirrors and lenses inside the
telescope YAML is purely a storage detail; in code, you only ever talk
in stages. Stage 0 is the primary; the renderer walks stages in
ascending order.

.. note::

   Operations are available both as standalone functions in
   :mod:`iactrace.telescope.operations` and as convenience methods on
   the :class:`~iactrace.telescope.Telescope` class. The method form is
   recommended::

       # Recommended: method form
       telescope = telescope.apply_roughness(stage=0, sigma=24.0)

       # Alternative: function form
       from iactrace.telescope import operations as ops
       telescope = ops.apply_roughness(telescope, stage=0, sigma=24.0)

Generic operations
------------------

These work on any kind of stage (mirror, lens, or slab).

**Surface Roughness**

Roughness broadens the PSF by perturbing surface normals at intersection
time. Specified in arcseconds RMS:

.. code-block:: python

   telescope = telescope.apply_roughness(stage=0, sigma=24.0)

**Misalignment**

Random Gaussian tip/tilt on element orientations (arcseconds):

.. code-block:: python

   import jax
   key = jax.random.key(42)

   telescope = telescope.apply_misalignment(
       stage=0,
       sigma_h=10.0,   # horizontal tip
       sigma_v=10.0,   # vertical tilt
       key=key,
   )

**Position and Orientation**

Set element positions and orientations directly:

.. code-block:: python

   import jax.numpy as jnp

   current_pos = telescope.stage(0).positions
   new_pos = current_pos.at[:, 2].add(0.001)  # shift +1mm in z
   telescope = telescope.set_positions(stage=0, positions=new_pos)

   new_rot = jnp.zeros_like(telescope.stage(0).rotations)
   telescope = telescope.set_rotations(stage=0, rotations=new_rot)

**Displacement**

Random Gaussian z-displacement (e.g. dish errors):

.. code-block:: python

   telescope = telescope.apply_displacement(
       stage=0, sigma_z=0.001, key=key,
   )

**Surface parameters**

Curvature, conic constant, aspheric coefficients on the
:class:`AsphericSurfaceGroup` underlying each stage:

.. code-block:: python

   n = telescope.stage(0).n_elements

   telescope = telescope.set_curvatures(
       stage=0, curvatures=jnp.full(n, 1.0 / 30.0),
   )
   telescope = telescope.scale_curvatures(stage=0, factor=1.01)
   telescope = telescope.offset_curvatures(stage=0, offset=0.001)

   telescope = telescope.set_conics(stage=0, conics=jnp.full(n, -1.0))
   telescope = telescope.set_aspherics(stage=0, aspherics=jnp.zeros((n, 4)))

   telescope = telescope.apply_conic_error(
       stage=0, sigma=0.01, key=key,
   )
   telescope = telescope.apply_aspheric_error(
       stage=0,
       sigmas=jnp.array([1e-6, 1e-8, 1e-10, 1e-12]),
       key=key,
   )

**Resample Monte-Carlo**

Refresh the sampling key on a single stage:

.. code-block:: python

   telescope = telescope.resample(stage=0, key=jax.random.key(7))

Kind-specific operations
------------------------

These validate the kind at the requested stage and raise
``ValueError`` if applied to the wrong kind.

**Reflectivity (mirror only)**

.. code-block:: python

   telescope = telescope.set_reflectivity(stage=0, reflectivity=0.95)
   telescope = telescope.scale_reflectivity(stage=0, factor=0.9)

**Transmittance (lens or slab only)**

.. code-block:: python

   telescope = telescope.set_transmittance(stage=2, transmittance=0.98)
   telescope = telescope.scale_transmittance(stage=2, factor=0.95)

**Refractive index (lens or slab only)**

.. code-block:: python

   telescope = telescope.set_refractive_index(stage=2, n_inside=1.52)

**Slab thickness (slab only)**

.. code-block:: python

   telescope = telescope.set_thickness(stage=2, thickness=0.005)

**Focal length (mirror or lens, kind-dispatched formula)**

For mirrors, ``c = 1 / (2 f)``. For single-surface refractive lenses,
``c = 1 / ((n_inside - n_outside) f)``, where ``n_outside`` is a design-time
ambient-index assumption passed to the operation (default ``1.0``; the lens
itself stores no ambient index -- the render loop reads it dynamically from
each ray's current medium). Slabs raise.

.. code-block:: python

   telescope = telescope.set_focal_lengths(
       stage=0, focal_lengths=jnp.full(n, 15.0),
   )

   # Manufacturing tolerances
   telescope = telescope.apply_focal_error(
       stage=0, sigma=0.01, key=key, relative=True,
   )

Camera frame operations
-----------------------

The telescope carries the camera frame (where rays are delivered to the
:class:`~iactrace.Camera`). The sensors themselves live on the camera
and are configured separately.

.. code-block:: python

   telescope = telescope.focus(delta_z=-0.005)
   telescope = telescope.set_camera_position(jnp.array([0.0, 0.0, 14.95]))
   telescope = telescope.set_camera_rotation(jnp.array([0.0, 1.0, 0.0]))

Sensor-side adjustments live on the camera:

.. code-block:: python

   camera = camera.set_sensor_positions(sensor_idx=0, positions=new_pos)
   camera = camera.set_sensor_rotations(sensor_idx=0, rotations=new_rot)

Obstruction operations
----------------------

Manage mechanical structures that block rays:

.. code-block:: python

   from iactrace.core import CylinderGroup

   mast = CylinderGroup(
       p1=jnp.array([[0.0, 0.0, 0.0]]),
       p2=jnp.array([[0.0, 0.0, 15.0]]),
       r=jnp.array([0.05]),
   )
   telescope = telescope.add_obstruction(mast)
   telescope = telescope.remove_obstruction(group_idx=0)
   telescope = telescope.clear_obstructions()

Stage discovery and summary
---------------------------

.. code-block:: python

   for s in telescope.stage_indices():
       g = telescope.stage(s)
       print(f"stage {s}: {g.kind}, {g.n_elements} elements")

   mirror_stages = telescope.stages_of_kind("mirror")
   lens_stages   = telescope.stages_of_kind("lens")

   info = telescope.get_info()
   print(f"Telescope: {info['name']}")
   print(f"  {info['n_mirror_elements']} mirror elements")
   print(f"  {info['n_lens_elements']} lens elements")
   print(f"  {info['n_obstructions']} obstructions")

Chaining operations
-------------------

.. code-block:: python

   key1, key2 = jax.random.split(key)

   telescope = (
       telescope
       .apply_roughness(stage=0, sigma=24.0)
       .apply_misalignment(stage=0, sigma_h=10.0, sigma_v=10.0, key=key1)
       .apply_focal_error(stage=0, sigma=0.01, key=key2, relative=True)
       .focus(-0.005)
   )

Next Steps
----------

- :doc:`/examples/index` — detailed examples for specific use cases
- :doc:`/api/telescope` — full API reference