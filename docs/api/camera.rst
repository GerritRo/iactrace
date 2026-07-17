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

Each :class:`~iactrace.camera.SensorGroup` owns a ``DetectionChain``:
an optional concentrator, a ``gap``, and a photodetector, applied to every
pixel of that group. Distinct groups in one camera can therefore carry
different cones or photodetectors. Configure a group's chain at construction
(``concentrator`` / ``photodetector`` / ``gap`` arguments) or functionally via
the ``sensor_idx``-keyed :meth:`Camera.set_concentrator`,
:meth:`Camera.set_photodetector`, and :meth:`Camera.set_gap` (documented on
:class:`~iactrace.camera.Camera` above).

A ray reaching a pixel is traced onto the photodetector's
:class:`~iactrace.camera.detector.surface.DetectionSurface` (its photocathode
geometry) and then weighted by the photodetector's response.

.. autoclass:: iactrace.camera.DetectionChain
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.detector.surface.DetectionSurface
   :members:
   :undoc-members:

Concentrators
-------------

Optional light concentrators (e.g. Winston cones) sit between the incoming
rays and the photodetector. Optical path length through the guide is weighted
by the concentrator's fill index (``1.0`` for an air-filled cone). All cones
share the :class:`~iactrace.camera.PolygonalCone` wall-tracing base.

.. autoclass:: iactrace.camera.Concentrator
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.PolygonalCone
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.camera.WinstonCone
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.camera.OkumuraCone
   :members:
   :undoc-members:
   :show-inheritance:

Photodetectors
--------------

A photodetector is the terminal element of a detection chain: it owns its
sensor surface and weights each landed ray by its detection efficiency.

.. autoclass:: iactrace.camera.PhotoDetector
   :members:
   :undoc-members:

.. autoclass:: iactrace.camera.ConstantQE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.camera.PMT
   :members:
   :undoc-members:
   :show-inheritance: