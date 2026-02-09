Features
========

This page provides an overview of IACTrace's capabilities.

Differentiable Ray Tracing
--------------------------

The core feature of IACTrace is end-to-end differentiability. Every operation
from light source to camera image can be differentiated using JAX's automatic
differentiation:

.. code-block:: python

   import jax
   import equinox as eqx
   from iactrace import Telescope

   # Define a loss function
   def loss_fn(telescope, sources, target_image):
       pred_image = telescope.render(sources, values, source_type='parallel')
       return jnp.mean((pred_image - target_image)**2)

   # Compute gradients w.r.t. telescope parameters
   grad_fn = eqx.filter_grad(loss_fn)
   gradients = grad_fn(telescope, sources, target_image)

Multi-Stage Optical Systems
---------------------------

IACTrace supports multi-stage optical configurations:

- **Single-mirror systems**: Davies-Cotton or parabolic primaries (H.E.S.S.,
  VERITAS, MAGIC)
- **Two-mirror systems**: Schwarzschild-Couder designs (pSCT, ASTRI)
- **Additional stages**: Windows, Lenses

Mirrors are organized into groups by optical stage, with each facet having
independent position, orientation, and surface parameters.

Aspheric Mirror Surfaces
------------------------

Mirror surfaces are described by the standard conic + polynomial form:

.. math::

   z(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}} + \sum_{i} A_{2i} r^{2i}

where :math:`c` is the curvature, :math:`k` is the conic constant, and
:math:`A_{2i}` are aspheric coefficients. This parameterization covers:

- Spherical mirrors (:math:`k = 0`)
- Parabolic mirrors (:math:`k = -1`)
- Hyperbolic mirrors (:math:`k < -1`)
- General aspherics with polynomial corrections

Sensor Types
------------

Two camera geometries are supported:

**Square sensors**
   Rectangular pixel grids with configurable resolution and physical bounds.
   Suitable for SiPM-based cameras or simulating images of lid cameras.

**Hexagonal sensors**
   Hexagonally-packed pixels matching the geometry of PMT-based IACT cameras.
   Proper handling of hexagon rotations.

Both sensor types support "straight-through" variants that have a soft backwards pass
via bilinear/barycentric interpolation

Obstruction Modeling
--------------------

Shadowing obstructions can be modeled via primitives, including

- **Cylinders**: Support struts, mast structures
- **Boxes**: Camera housings, electronics enclosures
- **Spheres**: Secondary mirror supports, actuator mechanisms
- **Triangles**: Triangle meshes for complicated shapes

Error Models
------------

Realistic telescope imperfections can be applied:

**Surface roughness**
   Micro-scale surface errors modeled as angular scattering with configurable
   RMS (in arcseconds).

**Mirror misalignment**
   Random tip/tilt errors for each facet, simulating alignment tolerances.
   Horizontal and vertical components can have different magnitudes.

**Position displacement**
   Axial (focus) errors in mirror positions, simulating mounting tolerances or
   thermal effects.

**Focal length errors**
   Variation in individual facet focal lengths from manufacturing tolerances.

**Conic and aspheric errors**
   Perturbations to higher-order surface parameters.

... and more.

YAML Configuration
------------------

Telescope geometries are defined in human-readable YAML files:

.. code-block:: yaml

   telescope:
     name: example_telescope
     units: m

   mirror_templates:
     primary:
       surface:
         curvature: 0.0667   # 1/15m focal length
         conic: -1.0         # Parabolic
         aspheric: []

   mirrors:
     - position: [0, 0, 0]
       orientation: [0, 0, 0]
       aperture:
         type: hexagonal
         radius: 0.3
       template: primary

   sensors:
     - type: hexagonal
       position: [0, 0, 15.0]
       n_pixels: 1855
       pixel_size: 0.05

Templates allow sharing surface parameters across multiple facets, reducing
configuration file size for large segmented mirrors.

Response Matrix Calculation
---------------------------

For calibration and analysis, IACTrace can compute the full source-to-pixel
response matrix:

.. code-block:: python

   from iactrace.core import render_response_matrix

   # Grid of source directions
   response = render_response_matrix(
       telescope, sources, values,
       source_type='parallel', sensor_idx=0
   )
   # response.shape = (n_sources, n_pixels)

This matrix encodes the effective area and PSF for each pixel as a function of
source position. This is especially useful for generating telescope approximations
for `nyx <https://github.com/GerritRo/nyx>`_, used to render the night sky.

Performance
-----------

IACTrace makes us of `JAX <https://github.com/jax-ml/jax>`_ to increase performance due to:

- **JIT compilation**: Functions are compiled to optimized XLA code on first
  call, with subsequent calls running at native speed.

- **GPU acceleration**: Seamlessly runs on NVIDIA GPUs when JAX is configured
  with CUDA support.

- **Vectorization**: Operations are automatically vectorized across sources,
  rays, and mirror facets.