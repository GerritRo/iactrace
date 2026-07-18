Physics and Conventions
=======================

This page is the contract: the coordinate frame, units, photometry
model, and which gradients flow through the pipeline. Everything below
applies uniformly across the library.

Coordinate frame
----------------

Coordinate frame like in sim_telarray. Assuming the telescope is first
pointed towards magnetic north and then up to the zenith.

- **Z-axis** along the optical axis, pointing from the primary mirror
  toward the sky (positive Z is "up", toward incoming light).
- **X-axis** along the north-south axis (positive x towards north -- meaning down to the ground in normal operation)
- **Y-axis** along the west-east axis (positive y towards west -- meaning looking at the mirrors from the camera y points towards the right)

The origin is conventionally at the centre of the primary-mirror dish.
After optics tracing, rays are expressed in the **camera-local frame**
defined by ``Telescope.camera_position`` and ``Telescope.camera_rotation``;
the :class:`~iactrace.Camera` operates entirely in that frame.

Units
-----

================  ===========================================================
Quantity          Unit
================  ===========================================================
Distance          metres
Angle (Euler)     degrees, XYZ intrinsic order
Small angles      arcseconds (mirror roughness, misalignment, focal-error
                  perturbations)
Time / path       metres of optical path length (``RayBundle.path_length``)
================  ===========================================================

There is no built-in unit system; the YAML configs and the runtime API
agree by convention only. Mixing units silently produces wrong results.

Photometry — monochromatic
--------------------------

**IACTrace is monochromatic.** Rays do not carry a wavelength. Every
coefficient that would in principle depend on wavelength is treated as a
single value:

- Mirror reflectivity (:class:`~iactrace.core.ReflectInteraction`)
- Lens refractive index and bulk transmittance
  (:class:`~iactrace.core.RefractInteraction`,
  :class:`~iactrace.core.SlabInteraction`)
- Photodetector quantum efficiency (:class:`~iactrace.camera.ConstantQE`,
  :class:`~iactrace.camera.PMT`)

This means PSF, throughput, and effective aperture results are
correctly differentiable and physically sensible *for a single
wavelength* — typically the photon-detection-weighted mean of whatever
spectrum you have in mind. Anything that requires a real spectrum
(wavelength-dependent QE for PMTs/SiPMs, dispersion through refractive
optics, reflectivity rolloff in the UV) is not modelled.

Wavelength tracking is planned for a future release.

The throughput-weighted values pipeline
---------------------------------------

``RayBundle.values`` is a dimensionless scalar per ray that accumulates
every multiplicative factor along the optical path::

    primary sampling weight
        × reflectivity / refractivity
        × aperture mask
        × obstruction shadow
        × concentrator throughput
        × quantum efficiency

By the time a bundle reaches :meth:`Camera.collect`, the entries of
``values`` are photoelectrons (per source-photon, per Monte-Carlo
sample). :meth:`Camera.image` and :meth:`Camera.response_matrix` sum
those values into pixel bins, so their output has the same units.

Two ways to drive the optics
----------------------------

There are two entry points; pick based on whether your source has a
closed-form ray sampler.

:meth:`Telescope.render` *is the fast path for point and parallel
sources.* It samples rays *backwards* from the primary aperture toward
each source — the only two cases where this is closed-form — and fuses
the per-mirror-element scan, so :meth:`Camera.image` and
:meth:`Camera.response_matrix` can fold over the full ray buffer
without ever materialising it.

:meth:`Telescope.trace` *is the general path.* You hand it raw
``(origins, directions, values)`` arrays and it propagates them
through. Use it for anything that ``render`` doesn't cover:

- extended sources (Gaussian, disk, image-as-source)
- arbitrary calibration ray patterns (collimated test beams,
  flashers with measured emission profiles)

There is no special source-type enum for these — sample whatever
distribution you want yourself, and call ``trace``.

Differentiability
-----------------

Ray tracing is end-to-end differentiable for *continuous* parameters.
Specifically, ``jax.grad`` flows through:

- mirror surface parameters: ``curvatures``, ``conics``, ``aspherics``,
  positions, rotations
- mirror reflectivity, lens refractive index and transmittance
- photodetector QE
- source positions / directions and source values

What does **not** currently flow:

- gradients through pixel bin assignment.
  :meth:`SensorGroup.pixel_index_and_mask` uses integer ``floor`` to
  assign each ray to a pixel, and :meth:`SensorGroup.scatter` bins the
  values with ``segment_sum``. Gradients flow w.r.t. the ray *values*
  (so ``d(image)/d(reflectivity)`` etc. are fine), but not through the
  bin *index* itself. Optimisations that need a continuous response to
  pixel boundaries (e.g. sensor-position fitting via image gradients)
  would need a straight-through estimator, which IACTrace does not
  currently ship.

For gradient-based work, prefer objectives built from the ray *values*
or from :doc:`focal-surface </api/analysis>` spot statistics, both of
which are fully differentiable, over ones that depend on which pixel a
ray falls in.

Optical stages
--------------

A telescope is a sequence of :class:`OpticalElementGroup` instances,
one per integer ``optical_stage``. Stage 0 is the primary; the
renderer walks stages in ascending integer order. Per-stage operations
are addressed by stage:

.. code-block:: python

   telescope.stage(0)             # OpticalElementGroup at stage 0
   telescope.stage(0).kind        # "mirror" | "lens" | "slab"
   telescope.stage_indices()      # sorted list of stages present
   telescope.stages_of_kind("mirror")

The split between ``mirror_groups`` and ``lens_groups`` on the
:class:`~iactrace.Telescope` is purely a storage layer that mirrors the
``mirrors:`` / ``lenses:`` sections of the YAML config; it has no
runtime semantics. The renderer consumes the combined
``optical_groups`` view sorted by stage.

The "one group per stage" rule is enforced at construction time. It is
the load-bearing invariant for ``stage(n)``.

Configuration files
-------------------

Telescope and camera live in separate YAML files; see
:doc:`/getting_started/installation` for setup and
:doc:`custom_telescopes` for the schema.

- ``Telescope.from_yaml`` returns a :class:`~iactrace.Telescope`. It
  carries the optics (mirrors, lenses, obstructions) plus the camera
  *frame* (position and orientation of the detector plane).
- ``Camera.from_yaml`` returns a :class:`~iactrace.Camera` whose
  sensor positions are interpreted in the camera-local frame.

The pairing happens at runtime, so a shared camera (e.g.
``FlashCAM.yaml``) can sit on multiple telescopes.