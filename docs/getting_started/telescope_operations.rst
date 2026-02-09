Telescope Operations
====================

IACTrace provides a functional API for modifying telescopes. All operations
return **new telescope instances** rather than modifying in place, enabling
reproducible simulations and compatibility with JAX transformations.

.. note::

   Operations are available both as standalone functions in
   :mod:`iactrace.telescope.operations` and as convenience methods on the
   :class:`~iactrace.telescope.Telescope` class. The method form is
   recommended for most use cases::

       # Recommended: method form
       telescope = telescope.apply_roughness(24.0)

       # Alternative: function form
       from iactrace.telescope.operations import apply_roughness
       telescope = apply_roughness(telescope, 24.0)


Mirror Operations
-----------------

**Surface Roughness**

Surface roughness broadens the point spread function (PSF) by adding random
angular perturbations to reflected rays. Values are specified in arcseconds
RMS:

.. code-block:: python

   # Apply 24 arcsec roughness to all mirrors
   telescope = telescope.apply_roughness(24.0)

   # Apply roughness to a specific mirror group only
   telescope = telescope.apply_roughness_to_group(group_idx=0, roughness=24.0)

**Mirror Misalignment**

Simulate alignment errors with Gaussian-distributed angular offsets:

.. code-block:: python

   import jax

   key = jax.random.key(42)

   # Apply random misalignment to primary mirrors (group 0)
   # sigma_h: horizontal (tip), sigma_v: vertical (tilt) in arcseconds
   telescope = telescope.apply_misalignment_to_group(
       group_idx=0,
       sigma_h=10.0,
       sigma_v=10.0,
       key=key
   )

**Position and Orientation**

Set mirror positions and orientations directly:

.. code-block:: python

   import jax.numpy as jnp

   # Get current positions
   current_pos = telescope.mirror_groups[0].positions

   # Modify positions (e.g., shift all mirrors by 1mm in z)
   new_pos = current_pos.at[:, 2].add(0.001)
   telescope = telescope.set_mirror_positions(group_idx=0, positions=new_pos)

   # Set rotations (Euler angles in degrees)
   new_rot = jnp.zeros_like(telescope.mirror_groups[0].rotations)
   telescope = telescope.set_mirror_rotations(group_idx=0, rotations=new_rot)

**Mirror Displacement**

Apply random z-axis displacement to simulate dish errors:

.. code-block:: python

   # Apply random z-displacement with 1mm standard deviation
   telescope = telescope.apply_displacement_to_group(
       group_idx=0,
       sigma_z=0.001,  # same units as positions (typically meters)
       key=key
   )

**Reflectivity Scaling**

Modify mirror reflectivity to simulate degradation or coating variations:

.. code-block:: python

   # Scale reflectivity for all mirrors in group 0
   telescope = telescope.scale_mirror_weights(group_idx=0, scale_factors=0.9)

   # Per-mirror scaling
   n_mirrors = len(telescope.mirror_groups[0])
   factors = jax.random.uniform(key, (n_mirrors,), minval=0.85, maxval=0.95)
   telescope = telescope.scale_mirror_weights(group_idx=0, scale_factors=factors)


Surface Parameter Operations
----------------------------

These operations modify the optical surface parameters (curvature, conic
constant, aspheric coefficients) without requiring resampling.

**Focal Length and Curvature**

For spherical/parabolic mirrors, curvature ``c = 1/(2f)`` where ``f`` is
the focal length:

.. code-block:: python

   # Set focal lengths directly
   focal_lengths = jnp.full(n_mirrors, 15.0)  # 15m focal length
   telescope = telescope.set_focal_lengths(group_idx=0, focal_lengths=focal_lengths)

   # Or set curvatures directly
   curvatures = 1.0 / (2.0 * 15.0)  # c = 1/(2f)
   telescope = telescope.set_mirror_curvatures(
       group_idx=0,
       curvatures=jnp.full(n_mirrors, curvatures)
   )

   # Scale existing curvatures
   telescope = telescope.scale_mirror_curvatures(group_idx=0, scale_factors=1.01)

   # Add offset to curvatures
   telescope = telescope.offset_mirror_curvatures(group_idx=0, offsets=0.001)

**Focal Length Errors**

Simulate manufacturing tolerances with random focal length perturbations:

.. code-block:: python

   # Absolute error (1cm standard deviation)
   telescope = telescope.apply_focal_error_to_group(
       group_idx=0,
       sigma=0.01,
       key=key,
       relative=False
   )

   # Relative error (1% standard deviation)
   telescope = telescope.apply_focal_error_to_group(
       group_idx=0,
       sigma=0.01,
       key=key,
       relative=True
   )

**Conic Constants**

Set or perturb conic constants (``k = -1`` for parabolic, ``k = 0`` for
spherical, ``k < -1`` for hyperbolic):

.. code-block:: python

   # Set conic constants
   conics = jnp.full(n_mirrors, -1.0)  # parabolic
   telescope = telescope.set_mirror_conics(group_idx=0, conics=conics)

   # Apply random conic errors
   telescope = telescope.apply_conic_error_to_group(
       group_idx=0,
       sigma=0.01,
       key=key
   )

**Aspheric Coefficients**

Higher-order surface corrections via aspheric polynomial terms:

.. code-block:: python

   # Set aspheric coefficients (N mirrors, K terms)
   aspherics = jnp.zeros((n_mirrors, 4))  # 4 aspheric terms
   telescope = telescope.set_mirror_aspherics(group_idx=0, aspherics=aspherics)

   # Apply random errors to aspheric terms
   sigmas = jnp.array([1e-6, 1e-8, 1e-10, 1e-12])  # per-term sigmas
   telescope = telescope.apply_aspheric_error_to_group(
       group_idx=0,
       sigmas=sigmas,
       key=key
   )


Sensor Operations
-----------------

**Focus Adjustment**

Move sensors along the optical axis (z-axis) for focusing:

.. code-block:: python

   # Move sensor 5mm closer to mirrors
   telescope = telescope.focus(delta_z=-0.005, sensor_idx=0)

**Adding and Removing Sensors**

.. code-block:: python

   from iactrace import SquareSensorGroup

   # Create a new sensor
   new_sensor = SquareSensorGroup(
       positions=jnp.array([[0.0, 0.0, 15.0]]),
       rotations=jnp.array([[0.0, 0.0, 0.0]]),
       width=100,
       height=100,
       bounds=(-0.5, 0.5, -0.5, 0.5),
   )

   # Add sensor
   telescope = telescope.add_sensor(new_sensor)

   # Replace existing sensor
   telescope = telescope.replace_sensor(new_sensor, idx=0)

   # Remove sensor
   telescope = telescope.remove_sensor(idx=1)

**Sensor Position and Rotation**

.. code-block:: python

   # Set sensor positions
   new_pos = jnp.array([[0.0, 0.0, 14.95]])
   telescope = telescope.set_sensor_positions(idx=0, positions=new_pos)

   # Set sensor rotations (Euler angles in degrees)
   new_rot = jnp.array([[0.0, 1.0, 0.0]])  # 1 degree tilt
   telescope = telescope.set_sensor_rotations(idx=0, rotations=new_rot)

**Straight-Through Estimator**

Convert sensors for gradient-based optimization:

.. code-block:: python

   # Convert to straight-through estimator for differentiable rendering
   telescope = telescope.with_ste(sensor_idx=0)


Obstruction Operations
----------------------

Manage mechanical structures that block rays:

.. code-block:: python

   from iactrace.core import CylinderObstructionGroup, BoxObstructionGroup

   # Add a cylindrical obstruction (e.g., camera support mast)
   mast = CylinderObstructionGroup(
       positions=jnp.array([[0.0, 0.0, 7.5]]),
       rotations=jnp.array([[0.0, 0.0, 0.0]]),
       radii=jnp.array([0.05]),
       half_lengths=jnp.array([7.5]),
   )
   telescope = telescope.add_obstruction(mast)

   # Remove an obstruction group by index
   telescope = telescope.remove_obstruction(group_idx=0)

   # Clear all obstructions
   telescope = telescope.clear_obstructions()


Utility Operations
------------------

**Getting Telescope Information**

.. code-block:: python

   info = telescope.get_info()
   print(f"Telescope: {info['name']}")
   print(f"Mirrors: {info['n_mirrors']} in {info['n_mirror_groups']} groups")
   print(f"Sensors: {info['n_sensors_total']} ({info['sensor_types']})")
   print(f"Obstructions: {info['n_obstructions']}")

**Cloning Telescopes**

Create independent copies.

.. code-block:: python

   # Create a copy
   telescope_copy = telescope.clone()

   # Modify the copy independently
   telescope_copy = telescope_copy.apply_roughness(30.0)

**Querying Mirror Groups**

.. code-block:: python

   # Get mirror groups by optical stage
   primary_indices = telescope.get_mirrors_by_stage(stage=0)
   secondary_indices = telescope.get_mirrors_by_stage(stage=1)

   # Count mirrors
   total_mirrors = telescope.get_mirror_count()


Chaining Operations
-------------------

Operations can be chained for complex modifications:

.. code-block:: python

   key1, key2, key3 = jax.random.split(key, 3)

   telescope = (
       telescope
       .apply_roughness(24.0)
       .apply_misalignment_to_group(0, 10.0, 10.0, key1)
       .apply_focal_error_to_group(0, 0.01, key2, relative=True)
       .focus(-0.005)
   )


Next Steps
----------

- :doc:`/examples/index` - Detailed examples for specific use cases
- :doc:`/api/telescope` - Full API reference for telescope operations