Custom Telescopes
=================

This guide explains how to create telescope and camera configurations for
your own optical systems.

Two YAML files
--------------

A telescope and its camera are described in **two separate YAML files**.
The telescope file owns the optics and the camera frame; the camera file
owns the sensor layout and the photodetector model. This split lets a single
shared camera file be paired with several telescope files (for example
``configs/CTAO/LST_camera.yaml`` is reused by all four LSTs).

.. code-block:: text

   my_telescope.yaml   <- mirrors, lenses, obstructions, camera frame
   my_camera.yaml      <- sensors, quantum efficiency, concentrator

Telescope YAML structure
------------------------

.. code-block:: yaml

   telescope:
     name: my_telescope
     units: m              # Length units (currently only "m")
     camera_position:      # Camera origin in world coordinates
       - 0.0
       - 0.0
       - 15.0
     camera_rotation:      # Camera orientation as Euler angles (degrees)
       - 0.0
       - 0.0
       - 0.0

   mirror_templates:
     # Surface parameter templates referenced by mirrors

   mirrors:
     # List of mirror facets

   lenses:
     # Optional list of refractive elements

   obstructions:
     # Optional list of shadow-casting structures

``camera_position`` and ``camera_rotation`` are required: rays are
transformed into this frame after tracing so that the camera works in its
own coordinate system.

Basic Example
-------------

A simple single-mirror telescope:

.. code-block:: yaml

   telescope:
     name: simple_parabolic
     units: m
     camera_position: [0.0, 0.0, 15.0]
     camera_rotation: [0.0, 0.0, 0.0]

   mirror_templates:
     primary:
       surface:
         curvature: 0.0333    # 1/(2*focal_length) for parabola
         conic: -1.0          # Parabolic
         aspheric: []

   mirrors:
     - position: [0, 0, 0]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 6.0
       template: primary

Mirror Definitions
------------------

Each mirror entry specifies:

**position** (required)
   ``[x, y, z]`` coordinates of the facet center.

**orientation** (required)
   ``[rx, ry, rz]`` Euler angles in degrees.

**aperture** (required)
   Shape of the mirror facet:

   .. code-block:: yaml

      # Circular aperture
      aperture:
        type: circular
        radius: 0.3

      # Circular with central hole
      aperture:
        type: circular
        radius: 0.3
        inner_radius: 0.05

      # Polygonal aperture (convex)
      aperture:
        type: polygon
        vertices: [[x1,y1],[x2,y2],[x3,y3],...,[xN,yN]]

**curvature, conic, aspheric, zernike, bsdf, reflectivity, coating** (all optional)
   A mirror is self-contained: it can set any of these directly, with no
   ``template`` at all. A field left unset defaults to flat / unmodified
   surface, perfect specular reflection, and reflectivity ``1.0`` -- see
   `Mirror Templates`_ for how ``template`` fills these in instead.

**template** (optional)
   Reference to a ``mirror_templates`` entry supplying defaults for
   whichever of the fields above the mirror itself leaves unset.

**stage** (optional)
   Optical stage index. Default is 0 (primary). Set to 1 for secondary
   mirrors, 2 for tertiary, etc. Each optical stage may contain at most
   one mirror group or lens group.

**id** (optional)
   Unique identifier for the facet.

Mirror Templates
----------------

A template supplies *defaults*, not requirements: every field it can set
(``surface``, ``bsdf``, ``reflectivity``, ``coating``) can also be set
directly on a mirror, and the mirror's own value always wins when both are
defined. A mirror is the **joint** of itself and its (optional) template,
resolved field by field -- not a fixed split between "shared" and
"per-mirror" data.

This makes segmented mirrors (many facets sharing most parameters, with one
that varies per facet) natural: put the shared parameters in a template and
override just the varying one on each facet.

.. code-block:: yaml

   mirror_templates:
     primary_facet:
       surface:
         curvature: 0.0333
         conic: -1.0
         aspheric: []

   mirrors:
     - position: [0, 0.6, 0]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 0.3
       template: primary_facet

     - position: [0.52, 0.3, 0]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 0.3
       template: primary_facet
       curvature: 0.0335   # this one facet's curvature overrides the template

     # ... more facets with same template

A facet's own ``zernike`` is a common use for this: share a template's
aspheric base across every panel, but give each panel its own measured
figure error --

.. code-block:: yaml

   mirrors:
     - position: [0, 0.6, 0]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 0.3
       template: primary_facet
       zernike:
         coeffs: [0.0, 0.0, 0.0, 0.0, 1.2e-4]  # this panel's measured astigmatism
         r_norm: 0.3

Lens Definitions (optional)
---------------------------

Refractive elements live under the top-level ``lenses:`` key. Two lens
types are supported:

Lenses share the same ``aperture`` block as mirrors -- ``circular`` and
``polygon`` apertures are both supported, so refractive elements can be
rectangular, hexagonal, etc.

**Aspheric disk** (curved refractive surface):

.. code-block:: yaml

   lenses:
     - type: aspheric_disk
       position: [0, 0, 14.5]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 0.5
       curvature: 0.05
       conic: 0.0
       n_inside: 1.5
       transmittance: 0.95  # default 1.0
       stage: 1

**Plano slab** (parallel-faced window):

.. code-block:: yaml

   lenses:
     - type: plano_slab
       position: [0, 0, 14.0]
       orientation: [0, 0, 0]
       aperture:
         type: polygon
         vertices: [[-0.20, -0.15], [0.20, -0.15], [0.20, 0.15], [-0.20, 0.15]]
       thickness: 0.005
       n_inside: 1.5
       transmittance: 0.98
       stage: 2

Obstruction Definitions
-----------------------

Obstructions model mechanical structures that block light:

**Cylinders** (support struts, masts):

.. code-block:: yaml

   obstructions:
     - type: cylinder
       p1: [0, 0, 0]         # Start point
       p2: [0, 0, 15]        # End point
       r: 0.05               # Radius

**Boxes (axis aligned)** (camera housings):

.. code-block:: yaml

   obstructions:
     - type: box
       p1: [-0.5, -0.5, 14.0]   # min corner
       p2: [ 0.5,  0.5, 14.5]   # max corner

**Spheres** (actuator mechanisms):

.. code-block:: yaml

   obstructions:
     - type: sphere
       center: [0, 0, 0.5]
       r: 0.1

Other available obstruction types are ``open_cylinder`` (open-ended
cylinder), ``oriented_box`` (rotated box; ``center`` + ``half_extents``
+ ``rotation``), and ``triangle`` (``v0``/``v1``/``v2`` vertices).

Camera YAML structure
---------------------

The camera file describes sensors in the **camera-local frame** (the
camera origin sits at the telescope's ``camera_position``):

.. code-block:: yaml

   camera:
     quantum_efficiency: 1.0
     # concentrator:           # optional
     #   type: hexagonal_cpc
     #   exit_inradius: 0.012
     #   acceptance_angle: 25.0

   sensors:
     - type: square
       position: [0, 0, 0]
       orientation: [0, 0, 0]
       width: 256              # pixels in X
       height: 256             # pixels in Y
       bounds: [-0.5, 0.5, -0.5, 0.5]   # [xmin, xmax, ymin, ymax]
       id: main_sensor

Sensor types are ``square`` (``width`` / ``height`` / ``bounds``) and
``hexagonal`` (``centers_x`` / ``centers_y`` lists of pixel centers).

Loading Custom Configurations
-----------------------------

Load the two halves with the public ``from_yaml`` constructors and pair
them at runtime:

.. code-block:: python

   import jax
   from iactrace import Telescope, Camera

   key = jax.random.key(0)

   telescope = Telescope.from_yaml(
       "my_telescope.yaml", n_samples=1024, key=key,
   )
   camera = Camera.from_yaml("my_camera.yaml")

Validating Configurations
-------------------------

After loading, inspect the telescope to verify it parsed correctly:

.. code-block:: python

   info = telescope.get_info()
   print(f"Mirror elements: {info['n_mirror_elements']}")
   print(f"Optical stages:  {info['n_stages']}")
   print(f"Obstructions:    {info['n_obstructions']}")

   # Visualize to check geometry
   from iactrace.viz import show_telescope
   scene = show_telescope(telescope)
   scene.show()