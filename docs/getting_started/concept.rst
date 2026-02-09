Concepts
========

This page explains the core concepts and terminology used in IACTrace.

Coordinate System
-----------------

IACTrace uses a right-handed Cartesian coordinate system:

- **X-axis**: Vertical, along the alt axis
- **Y-axis**: Horizontal, along the az axis
- **Z-axis**: Along the optical axis, pointing from the primary mirror toward
  the sky (positive Z is "up" toward incoming light)

.. note::

   The coordinate origin is typically at the center of the primary mirror dish.

All positions are specified in the units defined in the telescope configuration
(currently only supporting meters).

Angles are specified in **degrees** for orientations (Euler angles) and
**arcseconds** for small perturbations like surface roughness and misalignment.

Light Sources
-------------

IACTrace supports two types of light sources:

**Parallel sources** (``source_type='parallel'``)

Represent light from astronomical sources at effectively infinite distance.
The input is a direction vector ``[dx, dy, dz]`` pointing *toward* the source.
For on-axis light coming straight down onto the telescope:

.. code-block:: python

   direction = jnp.array([[0.0, 0.0, -1.0]])  # Light traveling in -Z direction

For an off-axis source at 1 degree in X:

.. code-block:: python

   angle_rad = 1.0 * jnp.pi / 180
   direction = jnp.array([[jnp.sin(angle_rad), 0.0, -jnp.cos(angle_rad)]])

**Point sources** (``source_type='point'``)

Represent light from finite-distance calibration sources (LEDs, flashers).
The input is a position ``[x, y, z]`` in the telescope coordinate system:

.. code-block:: python

   # LED at 200m distance, slightly off-axis
   position = jnp.array([[0.5, 0.3, 200.0]])

The telescope traces rays from this position to each mirror facet.

Monte Carlo Integration
-----------------------

For computational efficiency, IACTrace uses Monte-Carlo sampling for the primary
optical group when sampling rays. This removes the need for expensive intersection
computations on the first layer, at the cost of having to 'weight' rays with effective
ray apertures.

The ``MCIntegrator`` controls this sampling:

.. code-block:: python

   from iactrace import MCIntegrator

   integrator = MCIntegrator(n_samples=1024)

**n_samples**: Number of ray samples per mirror facet. Higher values give more
accurate results but increase computation time and memory usage. Recommended
values:

- Quick testing: 64-256
- Standard simulation: 1024-4096
- High precision: 8192+

The samples are drawn using JAX's PRNG system. Provide a random key when
loading the telescope to ensure reproducibility:

.. code-block:: python

   key = jax.random.key(42)  # Fixed seed for reproducibility
   telescope = load_telescope("config.yaml", integrator, key)

Telescope Structure
-------------------

A telescope in IACTrace consists of:

**Mirror groups**

Collections of mirror facets at the same optical stage. A single-mirror
telescope has one group (the primary). A Cassegrain has two groups (primary
and secondary).

Each facet in a group has:

- Position: ``[x, y, z]`` center location
- Orientation: ``[rx, ry, rz]`` Euler angles (degrees)
- Aperture: Shape and size (circular, hexagonal)
- Surface: Curvature, conic constant, aspheric terms

**Sensors**

Detector arrays at the focal plane. Each sensor has:

- Position and orientation in telescope coordinates
- Pixel geometry (square or hexagonal)
- Physical bounds or pixel size

**Obstructions**

Mechanical structures that block rays (support struts, camera housing).
Currently supports cylinders, boxes, and spheres.

Functional Operations
---------------------

IACTrace uses a functional programming style: operations return new telescope
instances rather than modifying in place. This design:

- Enables JAX's transformation system (``jit``, ``grad``, ``vmap``)
- Makes it easy to compare before/after states
- Avoids subtle mutation bugs

.. code-block:: python

   # Original telescope is unchanged
   original = load_telescope("config.yaml", integrator, key)

   # Operations return new instances
   with_roughness = original.apply_roughness(30.0)
   with_misalignment = with_roughness.apply_misalignment_to_group(0, 10.0, 10.0, key)

   # Can still use original
   image_perfect = original.render(sources, values, source_type='parallel')
   image_degraded = with_misalignment.render(sources, values, source_type='parallel')

Rendering vs Tracing
--------------------

IACTrace provides two main interfaces for simulation:

**render()**

High-level interface for simulating images from astronomical sources. Handles
the source-to-ray conversion internally:

.. code-block:: python

   image = telescope.render(sources, values, source_type='parallel')

**trace()**

Low-level interface for tracing explicit rays. You provide ray origins and
directions directly:

.. code-block:: python

   image = telescope.trace(ray_origins, ray_directions, ray_values)

Use ``trace()`` when you need precise control over ray geometry, such as
simulating LED calibration systems with known emission patterns. This also leads
to rays not having 'effective ray aperture' weighting.

Debug
-----

For debugging and analysis, both **trace()** and **render()** can use a debug flag
to pass ray coordinates without binning into pixels:

.. code-block:: python

   # Returns (positions, sensor_id, values) instead of binned image
   points, sensor_id, weights = telescope.trace(..., debug=True)

This is useful for:

- Visualizing the raw spot diagram
- Computing custom statistics on ray distributions
- Debugging optical alignment