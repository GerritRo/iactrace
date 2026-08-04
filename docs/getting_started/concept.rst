Concepts
========

This page introduces the moving parts. For the precise contract —
coordinates, units, photometry, differentiability — see
:doc:`conventions`.

Light Sources
-------------

IACTrace's ``render`` path supports two source types:

**Parallel sources** (``source_type='parallel'``)

Light from astronomical sources at effectively infinite distance. The
input is a direction vector ``[dx, dy, dz]`` pointing *toward* the
source. For on-axis light coming straight down onto the telescope:

.. code-block:: python

   direction = jnp.array([[0.0, 0.0, -1.0]])  # Light traveling in -Z direction

For an off-axis source at 1 degree in X:

.. code-block:: python

   angle_rad = 1.0 * jnp.pi / 180
   direction = jnp.array([[jnp.sin(angle_rad), 0.0, -jnp.cos(angle_rad)]])

**Point sources** (``source_type='point'``)

Light from finite-distance calibration sources (LEDs, flashers). The
input is a position ``[x, y, z]`` in the telescope coordinate system:

.. code-block:: python

   # LED at 200m distance, slightly off-axis
   position = jnp.array([[0.5, 0.3, 200.0]])

For anything else — extended sources, Cherenkov shower input, custom
calibration ray patterns — sample the rays yourself and use
:meth:`Telescope.trace` (see :doc:`conventions`).

Monte Carlo Integration
-----------------------

For computational efficiency, IACTrace uses Monte-Carlo sampling for
the primary optical group. This avoids expensive intersection
computations on the first surface, at the cost of having to weight
rays by an effective ray aperture.

The number of samples per mirror facet is set at load time via the
``n_samples`` argument to :meth:`Telescope.from_yaml`:

.. code-block:: python

   import jax
   from iactrace import Telescope

   key = jax.random.key(42)  # Fixed seed for reproducibility
   telescope = Telescope.from_yaml(
       "telescope.yaml", n_samples=1024, key=key,
   )

Higher values give more accurate results but increase compute and
memory. Pass a deterministic ``key`` to make the sampling reproducible.

Telescope Structure
-------------------

A telescope in IACTrace consists of:

**Mirror groups**

Collections of mirror facets at the same optical stage. A single-mirror
telescope has one group (the primary). A Cassegrain has two groups
(primary and secondary). Each facet in a group has:

- Position: ``[x, y, z]`` centre location
- Orientation: ``[rx, ry, rz]`` Euler angles (degrees)
- Aperture: shape and size (circular, hexagonal)
- Surface: curvature, conic constant, aspheric terms

**Obstructions**

Mechanical structures that block rays (support struts, camera
housing). Currently supports cylinders, boxes, and spheres.

**Camera frame**

Each telescope carries the position and Euler-angle orientation of the
camera (the ``camera_position`` / ``camera_rotation`` fields in the
telescope YAML). After tracing, rays are transformed into this local
camera frame so that the :class:`~iactrace.Camera` works in its own
coordinate system. Sensors, pixel layout and the photodetector model
live on the :class:`~iactrace.Camera` (loaded from a separate camera
YAML), not on the telescope.

Operations on a :class:`~iactrace.Telescope` return new instances
rather than mutating in place; see :doc:`telescope_operations` for the
full set and the rationale.

Rendering vs Tracing
--------------------

Both methods return a :class:`~iactrace.RayBundle` (or
:class:`~iactrace.LazyRayBundle`) in the camera-local frame; pass to
:meth:`Camera.image` for a binned pixel image, or
:meth:`Camera.collect` for per-ray output.

**render()** — high-level path for point and parallel sources. Handles
source-to-ray sampling internally and fuses the per-mirror-element
scan downstream:

.. code-block:: python

   ray_bundle = telescope.render(sources, values, source_type='parallel')
   image = camera.image(ray_bundle)

**trace()** — general path. You provide ray origins and directions
directly. Use this for any custom ray distribution that ``render``
doesn't cover (extended sources, Cherenkov shower output, calibration
patterns):

.. code-block:: python

   ray_bundle, trajectory = telescope.trace(ray_origins, ray_directions, ray_values)
   image = camera.image(ray_bundle)

``trace`` does not apply effective-aperture weighting because *you*
supplied the rays; it just propagates them. It returns a :class:`~iactrace.core.trajectory.TraceResult`, which is a NamedTuple with ``.rays`` being the ray bundle, and ``.trajectory`` holding the recorded path when called with ``record_trajectory=True``.

Raw ray output
--------------

For debugging and custom analysis, call :meth:`Camera.collect` instead
of :meth:`Camera.image` to get raw per-ray output without pixel
binning:

.. code-block:: python

   ray_bundle = telescope.render(sources, values, source_type='parallel')
   pe_vals, pe_times, pix_id, detected = camera.collect(ray_bundle)

``detected`` is the final per-ray mask: ``True`` only for rays that
stayed alive through the optics, hit a sensor tile, landed on the
photodetector surface, and fell inside a real pixel. ``pix_id`` and
``pe_times`` are meaningful only where ``detected`` is ``True``
(``pe_vals`` is already zeroed elsewhere). This is useful for:

- visualising the raw spot diagram
- computing custom statistics on ray distributions
- debugging optical alignment