API Reference
=============

This section provides detailed API documentation for all IACTrace modules.

Primary Interfaces
------------------

These modules contain the main user-facing APIs:

.. toctree::
   :maxdepth: 2

   iactrace
   telescope
   sensors
   viz

Core Components
---------------

Lower-level modules for advanced usage:

.. toctree::
   :maxdepth: 2

   core
   io
   utils

Telescope Operations
--------------------

The :mod:`iactrace.telescope.operations` module provides functional operations
for modifying telescopes. See :doc:`/getting_started/quickstart` for usage
examples.

Key operations include:

**Mirror modifications:**

- ``set_mirror_positions`` - Update facet positions
- ``set_mirror_rotations`` - Update facet orientations
- ``set_mirror_curvatures`` - Modify surface curvatures
- ``set_focal_lengths`` - Set focal lengths (computes curvature)
- ``resample_mirrors`` - Regenerate Monte Carlo sample points

**Error simulation:**

- ``apply_roughness`` - Add surface micro-roughness
- ``apply_misalignment_to_group`` - Random tip/tilt errors
- ``apply_displacement_to_group`` - Random axial displacement
- ``apply_focal_error_to_group`` - Focal length perturbations

**Sensor operations:**

- ``focus`` - Adjust sensor position along optical axis
- ``add_sensor`` - Add a new sensor
- ``set_sensor_position`` - Move sensor to specific location

**Obstruction operations:**

- ``add_obstruction`` - Add shadow-casting structure
- ``clear_obstructions`` - Remove all obstructions

**Utility functions:**

- ``get_info`` - Get telescope summary information
- ``clone`` - Create independent copy of telescope