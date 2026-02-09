iactrace.sensors
================

Sensor classes for detecting rays at the focal plane.

Base Class
----------

.. autoclass:: iactrace.sensors.SensorGroup
   :members:
   :undoc-members:

Standard Sensors
----------------

These sensors accumulate ray hits into pixel bins:

.. autoclass:: iactrace.sensors.SquareSensorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.sensors.HexagonalSensorGroup
   :members:
   :undoc-members:
   :show-inheritance:

Straight-Through Sensors
------------------------

These sensors pass ray coordinates through without binning, useful for
debugging, analysis, and optimization (cleaner gradients):

.. autoclass:: iactrace.sensors.StraightThroughSquareSensorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.sensors.StraightThroughHexagonalSensorGroup
   :members:
   :undoc-members:
   :show-inheritance: