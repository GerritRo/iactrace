iactrace.camera
===============

The Camera class and detector-side components.

Camera Class
------------

.. autoclass:: iactrace.camera.Camera
   :members:
   :undoc-members:

Sensor Groups
-------------

Sensors live in the camera-local frame and accumulate rays into pixels.

.. autoclass:: iactrace.camera.SensorGroup
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.SquareSensorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.camera.HexagonalSensorGroup
   :members:
   :undoc-members:
   :show-inheritance:

Concentrators
-------------

Optional light concentrators (e.g. Winston cones) sit between the
incoming rays and the photosensor:

.. autoclass:: iactrace.camera.Concentrator
   :members:
   :undoc-members:

Photosensors
------------

Photosensor models (quantum efficiency).

.. autoclass:: iactrace.camera.PhotoSensor
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.UniformQE
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Functional operations for modifying camera configurations.

.. automodule:: iactrace.camera.operations
   :members:
   :undoc-members:
