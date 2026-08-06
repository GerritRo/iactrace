iactrace
========

Top-level package providing convenient access to the most commonly used classes
and functions.

All items listed here are re-exported from their respective submodules for
convenience. See the linked module documentation for full details.

Quick Reference
---------------

Core Classes
^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.telescope.Telescope`
     - Optical system: mirrors, lenses, obstructions, camera frame
   * - :class:`~iactrace.camera.Camera`
     - Detection system: sensor groups, concentrator, photodetector
   * - :class:`~iactrace.core.RayBundle`
     - Ray positions/directions/values exchanged between the two
   * - :class:`~iactrace.core.LazyRayBundle`
     - Deferred render returned by :meth:`Telescope.render`, folded by the camera

Sensor Classes
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.camera.SquareSensorGroup`
     - Rectangular pixel grid sensor
   * - :class:`~iactrace.camera.HexagonalSensorGroup`
     - Hexagonally-packed pixel sensor

Photodetectors / Concentrators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.camera.ConstantQE`
     - Flat scalar quantum efficiency (the default detector)
   * - :class:`~iactrace.camera.PMT`
     - Photomultiplier: sensor surface, QE, optional Fresnel window
   * - :class:`~iactrace.camera.WinstonCone`
     - Winston light concentrator (per pixel)
   * - :class:`~iactrace.camera.OkumuraCone`
     - Okumura-style light concentrator (per pixel)

I/O Functions
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :meth:`Telescope.from_yaml <iactrace.telescope.Telescope.from_yaml>`
     - Load a telescope from a standalone telescope YAML file
   * - :meth:`Camera.from_yaml <iactrace.camera.Camera.from_yaml>`
     - Load a camera from a standalone camera YAML file
   * - :func:`~iactrace.io.save_telescope`
     - Save a telescope to a standalone YAML file
   * - :func:`~iactrace.io.save_camera`
     - Save a camera to a standalone YAML file

Visualization Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`~iactrace.viz.show_image`
     - Display a sensor image (hexagonal or square pixel layout)

Analysis Tools
^^^^^^^^^^^^^^

Available under :mod:`iactrace.analysis`:

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.analysis.FlatFocalPlane`
     - Flat focal plane for spot-diagram analysis
   * - :class:`~iactrace.analysis.AsphericFocalSurface`
     - Curved aspheric focal surface for spot-diagram analysis
   * - :class:`~iactrace.analysis.FocalSurfaceHits`
     - Per-ray intersection results returned by :meth:`FocalSurface.intersect`