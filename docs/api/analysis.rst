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

Effective Aperture
------------------

Tabulate what a telescope and camera together do to a plane wave: the
per-pixel effective area, over field angle and over wavelength. The scan is
the expensive step, so a table is normally written once and read back many
times -- see :func:`iactrace.io.save_aperture_table` for the format, which
numpy alone can read.

.. autofunction:: iactrace.analysis.effective_aperture

.. autoclass:: iactrace.analysis.EffectiveApertureTable
   :members:
   :undoc-members:

.. autoclass:: iactrace.analysis.FieldFrame
   :members:

.. autofunction:: iactrace.analysis.effective_area

.. autofunction:: iactrace.analysis.pixel_response
