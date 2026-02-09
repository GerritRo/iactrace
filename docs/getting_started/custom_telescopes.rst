Custom Telescopes
=================

This guide explains how to create telescope configurations for your own
optical systems.

YAML Configuration Structure
----------------------------

Telescope configurations are defined in YAML files with the following sections:

.. code-block:: yaml

   telescope:
     name: my_telescope
     units: m           # Length units

   mirror_templates:
     # Surface parameter templates (optional)

   mirrors:
     # List of mirror facets

   sensors:
     # List of detector arrays

   obstructions:
     # List of shadow-casting structures (optional)

Basic Example
-------------

A simple single-mirror telescope with a hexagonal sensor:

.. code-block:: yaml

   telescope:
     name: simple_parabolic
     units: m

   mirrors:
     - position: [0, 0, 0]
       orientation: [0, 0, 0]
       aperture:
         type: circular
         radius: 6.0
       surface:
         curvature: 0.0333    # 1/(2*focal_length) for parabola
         conic: -1.0          # Parabolic
         aspheric: []

   sensors:
     - type: square
       position: [0, 0, 15.0]
       orientation: [0, 0, 0]
       width: 256            # Pixels in X
       height: 256           # Pixels in Y
       bounds: [-0.5, 0.5, -0.5, 0.5]  # [xmin, xmax, ymin, ymax]

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

**surface** (required unless using template)
   Optical surface parameters:

   .. code-block:: yaml

      surface:
        curvature: 0.0333     # 1/radius of curvature
        conic: -1.0           # Conic constant (0=sphere, -1=parabola)
        aspheric: [0, 0]      # Higher-order coefficients [A4, A6, ...]

**stage** (optional)
   Optical stage index. Default is 0 (primary). Set to 1 for secondary mirrors,
   2 for tertiary, etc.

**id** (optional)
   Unique identifier for the facet.

**template** (optional)
   Reference to a mirror template for surface parameters.

Using Mirror Templates
----------------------

For segmented mirrors where many facets share surface parameters, use
templates to avoid repetition:

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

     # ... more facets with same template

Sensor Definitions
------------------

Sensor pixels have to be on a regular grid. Each sensor entry specifies:

**type** (required)
   Either ``square`` or ``hexagonal``.

**position** (required)
   ``[x, y, z]`` coordinates of the sensor center.

**orientation** (optional)
   ``[rx, ry, rz]`` Euler angles in degrees. Default ``[0, 0, 0]``.

For **square sensors**:

.. code-block:: yaml

   sensors:
     - type: square
       position: [0, 0, 15.0]
       orientation: [0, 0, 0]
       width: 256            # Pixels in X
       height: 256           # Pixels in Y
       bounds: [-0.5, 0.5, -0.5, 0.5]  # [xmin, xmax, ymin, ymax]

For **hexagonal sensors**:

.. code-block:: yaml

   sensors:
     - type: hexagonal
       position: [0, 0, 15.0]
       orientation: [0, 0, 0]
       centers_x: [x1,x2,x3,...,xN] # Hexagon centers in X
       centers_y: [y1,y2,y3,...,yN] # Hexagon centers in Y

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
       center: [0, 0, 14.5]
       size: [1.0, 1.0, 0.5]  # [width, height, depth]

**Spheres** (actuator mechanisms):

.. code-block:: yaml

   obstructions:
     - type: sphere
       center: [0, 0, 0.5]
       radius: 0.1

Loading Custom Configurations
-----------------------------

Load your configuration like any other:

.. code-block:: python

   from iactrace import MCIntegrator, load_telescope
   import jax

   telescope = load_telescope(
       "my_telescope.yaml",
       MCIntegrator(n_samples=1024),
       jax.random.key(0)
   )

Validating Configurations
-------------------------

After loading, inspect the telescope to verify it parsed correctly:

.. code-block:: python

   from iactrace.telescope.operations import get_info

   info = get_info(telescope)
   print(f"Mirrors: {info['n_mirrors']}")
   print(f"Sensors: {info['n_sensors']}")
   print(f"Optical stages: {info['optical_stages']}")

   # Visualize to check geometry
   from iactrace.viz import show_telescope
   scene = show_telescope(telescope)
   scene.show()