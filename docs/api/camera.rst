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

Detection chain
---------------

Each :class:`~iactrace.camera.SensorGroup` owns a ``DetectionChain``,
an optional concentrator, a ``gap``, and a photosensor, applied to every
pixel of that group. Distinct groups in one camera can therefore carry
different cones or photosensors. Configure a group's chain at construction
(``concentrator`` / ``photosensor`` / ``gap`` arguments) or functionally via
the ``sensor_idx``-keyed ``Camera.set_concentrator`` / ``set_photosensor`` /
``set_gap``.

.. autoclass:: iactrace.camera.DetectionChain
   :members:
   :undoc-members:

Concentrators
-------------

Optional light concentrators (e.g. Winston cones) sit between the
incoming rays and the photosensor. Optical path length through the guide is
weighted by the concentrator's fill index (1.0 for an air-filled Winston cone):

.. autoclass:: iactrace.camera.Concentrator
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.WinstonCone
   :members:
   :undoc-members:
   :show-inheritance:

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