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
     - Detection system: sensor groups, concentrator, photosensor
   * - :class:`~iactrace.core.RayBundle`
     - Ray positions/directions/values exchanged between the two

Sensor Classes
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.camera.SquareSensorGroup`
     - Rectangular pixel grid sensor
   * - :class:`~iactrace.camera.HexagonalSensorGroup`
     - Hexagonally-packed pixel sensor

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

   * - :func:`~iactrace.viz.hexshow`
     - Display hexagonal sensor image
   * - :func:`~iactrace.viz.squareshow`
     - Display square sensor image
