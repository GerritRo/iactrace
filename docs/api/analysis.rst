iactrace.analysis
=================

Analysis tools for post-processing ray bundles.

Focal Surface
-------------

Intersect a :class:`~iactrace.RayBundle` with a parametric focal surface
to inspect spot diagrams, chief-ray angles, and other PSF metrics
without going through the camera's pixel binning.

.. autoclass:: iactrace.analysis.FocalSurface
   :members:
   :undoc-members:

.. autoclass:: iactrace.analysis.FlatFocalPlane
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.analysis.AsphericFocalSurface
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.analysis.FocalSurfaceHits
   :members:
   :undoc-members:
