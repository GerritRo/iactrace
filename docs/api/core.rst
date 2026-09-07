iactrace.core
=============

Low-level ray tracing components. Most users should use the
:class:`~iactrace.telescope.Telescope` and :class:`~iactrace.camera.Camera`
classes instead of these functions directly.

Render Engine
-------------

``render_optics`` generates rays from sources and materialises the full ray
buffer; ``render_optics_accumulate`` folds an accumulator over primary
elements instead, so peak memory does not grow with element count.
``trace_optics`` takes caller-supplied rays.

.. autofunction:: iactrace.core.render_optics

.. autofunction:: iactrace.core.render_optics_accumulate

.. autofunction:: iactrace.core.trace_optics

Handoff to a local frame
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: iactrace.core.handoff_to_frame

.. autofunction:: iactrace.core.apply_final_leg_shadow

.. autofunction:: iactrace.core.final_leg_points

Ray Bundle
----------

.. autodata:: iactrace.core.DEFAULT_WAVELENGTH

.. autoclass:: iactrace.core.RayBundle
   :members:

.. autoclass:: iactrace.core.LazyRayBundle
   :members:

Trajectories
~~~~~~~~~~~~

.. autoclass:: iactrace.core.TraceResult
   :members:

.. autoclass:: iactrace.core.Trajectory
   :members:

Source Spectra
--------------

What a source emits, as a distribution over wavelength.

.. autoclass:: iactrace.core.Spectrum
   :members:

.. autoclass:: iactrace.core.ConstantSpectrum
   :show-inheritance:

.. autoclass:: iactrace.core.TabulatedSpectrum
   :members:
   :show-inheritance:

.. autofunction:: iactrace.core.as_spectrum

Refractive Index
----------------

A refracting element's index as a function of wavelength.

.. autoclass:: iactrace.core.RefractiveIndex
   :members:

.. autoclass:: iactrace.core.ConstantIndex
   :show-inheritance:

.. autoclass:: iactrace.core.TabulatedIndex
   :members:
   :show-inheritance:

.. autoclass:: iactrace.core.SellmeierIndex
   :show-inheritance:

.. autofunction:: iactrace.core.as_refractive_index

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

Response Curves
~~~~~~~~~~~~~~~

Angle- and wavelength-dependent reflectivity / transmittance applied at an
interaction.

.. autoclass:: iactrace.core.ResponseCurve
   :members:

.. autoclass:: iactrace.core.ConstantResponse
   :show-inheritance:

.. autoclass:: iactrace.core.TabulatedResponse
   :members:
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

.. autofunction:: iactrace.core.reflect

.. autofunction:: iactrace.core.refract

.. autofunction:: iactrace.core.refract_slab

.. autofunction:: iactrace.core.fresnel_unpolarized

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

.. autofunction:: iactrace.core.sag

.. autofunction:: iactrace.core.compute_sag_and_normal

.. autofunction:: iactrace.core.zernike_terms

.. autofunction:: iactrace.core.bicubic_interp

.. autofunction:: iactrace.core.sag_raw

.. autodata:: iactrace.core.N_ZERNIKE

Intersection Functions
----------------------

Geometric ray-primitive intersection tests (in
:mod:`iactrace.core.intersections`):

.. autofunction:: iactrace.core.intersect_plane

.. autofunction:: iactrace.core.intersect_sphere

.. autofunction:: iactrace.core.intersect_cylinder

.. autofunction:: iactrace.core.intersect_open_cylinder

.. autofunction:: iactrace.core.intersect_box

.. autofunction:: iactrace.core.intersect_oriented_box

.. autofunction:: iactrace.core.intersect_triangle

.. autofunction:: iactrace.core.intersect_conic

For a surface with no closed-form root, the generic Newton solver:

.. autofunction:: iactrace.core.newton_raphson_intersect

.. autofunction:: iactrace.core.is_hit

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

Aperture Sampling
-----------------

Uniform sampling over an element's aperture, used when a render generates its
primary-surface rays.

.. autofunction:: iactrace.core.sample_annulus

.. autofunction:: iactrace.core.sample_polygon

Numerical Tolerances
--------------------

Resolution-relative floors, derived from the working dtype.

.. autofunction:: iactrace.core.dir_tol

.. autofunction:: iactrace.core.len_rel
