iactrace.core
=============

Low-level ray tracing components. Most users should use the
:class:`~iactrace.telescope.Telescope` class instead of these functions
directly.

Rendering Functions
-------------------

High-level functions for rendering images through a telescope:

.. autofunction:: iactrace.core.render

.. autofunction:: iactrace.core.render_debug

.. autofunction:: iactrace.core.render_response_matrix

.. autofunction:: iactrace.core.trace_rays

.. autofunction:: iactrace.core.trace_rays_debug

Integrators
-----------

Classes for sampling rays on optical surfaces:

.. autoclass:: iactrace.core.Integrator
   :members:
   :undoc-members:

.. autoclass:: iactrace.core.MCIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

Optical Physics
---------------

Functions for ray-surface interactions:

.. autofunction:: iactrace.core.reflect

.. autofunction:: iactrace.core.refract

.. autofunction:: iactrace.core.refract_slab

.. autofunction:: iactrace.core.fresnel_unpolarized

Surfaces
--------

Aspheric surface calculations:

.. autoclass:: iactrace.core.AsphericSurface
   :members:
   :undoc-members:

.. autofunction:: iactrace.core.sag

.. autofunction:: iactrace.core.compute_sag_and_normal

Intersection Functions
----------------------

Geometric ray-primitive intersection tests:

.. autofunction:: iactrace.core.intersect_plane

.. autofunction:: iactrace.core.intersect_sphere

.. autofunction:: iactrace.core.intersect_cylinder

.. autofunction:: iactrace.core.intersect_box

.. autofunction:: iactrace.core.intersect_oriented_box

.. autofunction:: iactrace.core.intersect_triangle

.. autofunction:: iactrace.core.intersect_conic

Obstruction Groups
------------------

Classes for modeling ray obstructions:

.. autoclass:: iactrace.core.ObstructionGroup
   :members:
   :undoc-members:

.. autoclass:: iactrace.core.CylinderGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.core.BoxGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.core.SphereGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.core.OrientedBoxGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.core.TriangleGroup
   :members:
   :undoc-members:
   :show-inheritance:

Transforms
----------

Coordinate transformation utilities:

.. autofunction:: iactrace.core.euler_to_matrix

.. autofunction:: iactrace.core.look_at_rotation