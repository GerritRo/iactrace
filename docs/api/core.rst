iactrace.core
=============

Low-level ray tracing components. Most users should use the
:class:`~iactrace.telescope.Telescope` and :class:`~iactrace.camera.Camera`
classes instead of these functions directly.

Render Engine
-------------

.. autofunction:: iactrace.core.render_optics

.. autofunction:: iactrace.core.trace_optics

Ray Bundle
----------

.. autoclass:: iactrace.core.RayBundle
   :members:

Optical Element Composition
---------------------------

.. autoclass:: iactrace.core.OpticalElementGroup
   :members:

Apertures
~~~~~~~~~

.. autoclass:: iactrace.core.Aperture
   :members:

.. autoclass:: iactrace.core.DiskAperture
   :members:
   :show-inheritance:

.. autoclass:: iactrace.core.PolygonAperture
   :members:
   :show-inheritance:

Interactions
~~~~~~~~~~~~

.. autoclass:: iactrace.core.Interaction
   :members:

.. autoclass:: iactrace.core.ReflectInteraction
   :show-inheritance:

.. autoclass:: iactrace.core.RefractInteraction
   :show-inheritance:

.. autoclass:: iactrace.core.SlabInteraction
   :show-inheritance:

.. autoclass:: iactrace.core.InteractionType
   :members:

Coatings
~~~~~~~~

Angle-dependent reflectivity / transmittance applied at an interaction.

.. autoclass:: iactrace.core.Coating
   :members:

.. autoclass:: iactrace.core.ConstantCoating
   :show-inheritance:

.. autoclass:: iactrace.core.TabulatedCoating
   :show-inheritance:

BSDF (surface scattering)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: iactrace.core.BSDF
   :members:

.. autoclass:: iactrace.core.GaussianBSDF
   :show-inheritance:

.. autoclass:: iactrace.core.DoubleGaussianBSDF
   :show-inheritance:

Optical Physics
---------------

Functions for ray-surface interactions:

.. autofunction:: iactrace.core.interactions.reflect

.. autofunction:: iactrace.core.interactions.refract

.. autofunction:: iactrace.core.interactions.refract_slab

.. autofunction:: iactrace.core.coatings.fresnel_unpolarized

Surfaces
--------

Surface-figure models. ``SurfaceGroup`` is the base; the concrete groups
below can be combined with :class:`~iactrace.core.SumSurfaceGroup` (e.g. an
aspheric base plus a per-facet Zernike figure error).

.. autoclass:: iactrace.core.SurfaceGroup
   :members:
   
.. autoclass:: iactrace.core.SumSurfaceGroup
   :members:
   :show-inheritance:

.. autoclass:: iactrace.core.AsphericSurfaceGroup
   :members:
   :show-inheritance:

.. autoclass:: iactrace.core.ZernikeSurfaceGroup
   :members:
   :show-inheritance:

.. autoclass:: iactrace.core.FreeformSurfaceGroup
   :members:
   :show-inheritance:

.. autofunction:: iactrace.core.surfaces.sag

.. autofunction:: iactrace.core.surfaces.compute_sag_and_normal

.. autofunction:: iactrace.core.zernike_terms

.. autofunction:: iactrace.core.bicubic_interp

Intersection Functions
----------------------

Geometric ray-primitive intersection tests (in
:mod:`iactrace.core.intersections`):

.. autofunction:: iactrace.core.intersections.intersect_plane

.. autofunction:: iactrace.core.intersections.intersect_sphere

.. autofunction:: iactrace.core.intersections.intersect_cylinder

.. autofunction:: iactrace.core.intersections.intersect_open_cylinder

.. autofunction:: iactrace.core.intersections.intersect_box

.. autofunction:: iactrace.core.intersections.intersect_oriented_box

.. autofunction:: iactrace.core.intersections.intersect_triangle

.. autofunction:: iactrace.core.intersections.intersect_conic

Obstruction Groups
------------------

Classes for modeling ray obstructions:

.. autoclass:: iactrace.core.ObstructionGroup
   :members:

.. autoclass:: iactrace.core.CylinderGroup
   :show-inheritance:

.. autoclass:: iactrace.core.OpenCylinderGroup
   :show-inheritance:

.. autoclass:: iactrace.core.BoxGroup
   :show-inheritance:

.. autoclass:: iactrace.core.SphereGroup
   :show-inheritance:

.. autoclass:: iactrace.core.OrientedBoxGroup
   :show-inheritance:

.. autoclass:: iactrace.core.TriangleGroup
   :show-inheritance:

Transforms
----------

Coordinate transformation utilities:

.. autofunction:: iactrace.core.euler_to_matrix