"""Camera: pixelated sensors and the per-pixel detection chain.

Ray flow
--------
A camera-frame :class:`~iactrace.core.ray_bundle.RayBundle` (from
``Telescope.render`` / ``trace``) passes through one pipeline::

    intersect_sensor      camera frame -> sensor-tile frame (nearest tile)
    to_pixel_frame        sensor tile  -> pixel-local frame (assigned pixel)
    DetectionChain        concentrator.to_surface -> photodetector response
    scatter / collect     pixel binning (image) or per-ray output

Everything downstream of ``to_pixel_frame`` happens in **one frame**, the
pixel-local frame: the pixel entrance aperture spans ``z = 0`` with light
travelling toward ``-z``, a concentrator occupies ``z in [-length, 0]``, and
the detector plane sits at ``z = -(length + gap)``.

Who owns what
-------------
* :class:`SensorGroup` (square / hexagonal): pixel layout and binning; each
  group owns one :class:`DetectionChain` shared by all its pixels.
* :class:`DetectionChain`: composition only -- an optional
  :class:`Concentrator`, a mounting ``gap``, and a :class:`PhotoDetector`.
* :class:`Concentrator` (:class:`WinstonCone`, :class:`OkumuraCone`, both
  via :class:`PolygonalCone`): delivers rays onto the sensor surface through its
  own geometry via one polymorphic primitive,
  :meth:`~iactrace.camera.optics.concentrator.Concentrator.to_surface`. A
  wall-based cone does so by bouncing rays through the shared :func:`trace_chain`;
  a future lens concentrator would refract instead -- the pipeline stays agnostic.
* :class:`PhotoDetector` (:class:`ConstantQE`, :class:`PMT`): owns its sensor
  surface (``surface`` -> :class:`~iactrace.camera.detector.surface.DetectionSurface`,
  built on the core surface machinery) and its detection response
  (``detect``), including any angular dependence.

Loss accounting rides on the bundle itself: ``alive`` flips off on geometry
loss (missed every tile, lost in the cone, missed the photocathode) and
``values`` carry the radiometric throughput -- a dead ray always has
``values == 0`` and meaningless positions.
"""

from .camera import Camera
from .detection_chain import DetectionChain
from .detector import PMT, ConstantQE, PhotoDetector
from .optics import ChainTrace, Concentrator, OkumuraCone, PolygonalCone, WinstonCone, trace_chain
from .sensor_group import HexagonalSensorGroup, SensorGroup, SquareSensorGroup

__all__ = [
    "Camera",
    "DetectionChain",
    "Concentrator",
    "PolygonalCone",
    "WinstonCone",
    "OkumuraCone",
    "SensorGroup",
    "SquareSensorGroup",
    "HexagonalSensorGroup",
    "PhotoDetector",
    "PMT",
    "ConstantQE",
    "trace_chain",
    "ChainTrace",
]
