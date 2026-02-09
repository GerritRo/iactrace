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
     - Main telescope class for ray tracing simulations
   * - :class:`~iactrace.core.MCIntegrator`
     - Monte Carlo integrator for sampling rays on surfaces
   * - :class:`~iactrace.core.Integrator`
     - Base integrator class

Sensor Classes
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`~iactrace.sensors.SquareSensorGroup`
     - Rectangular pixel grid sensor
   * - :class:`~iactrace.sensors.HexagonalSensorGroup`
     - Hexagonally-packed pixel sensor
   * - :class:`~iactrace.sensors.StraightThroughSquareSensorGroup`
     - Square sensor returning raw ray coordinates
   * - :class:`~iactrace.sensors.StraightThroughHexagonalSensorGroup`
     - Hexagonal sensor returning raw ray coordinates

I/O Functions
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`~iactrace.io.load_telescope`
     - Load telescope from YAML configuration file
   * - :func:`~iactrace.io.save_telescope`
     - Save telescope to YAML configuration file
   * - :func:`~iactrace.io.build_telescope`
     - Build telescope from configuration dictionary

Visualization Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`~iactrace.viz.hexshow`
     - Display hexagonal sensor image
   * - :func:`~iactrace.viz.squareshow`
     - Display square sensor image