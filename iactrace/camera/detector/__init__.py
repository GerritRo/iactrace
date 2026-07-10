"""Photodetectors and the sensor surface they own.

:class:`~iactrace.camera.detector.surface.DetectionSurface` is the traced
photocathode geometry a detection chain delivers rays onto;
:class:`~iactrace.camera.detector.photodetector.PhotoDetector` (with
:class:`~iactrace.camera.detector.photodetector.ConstantQE` and
:class:`~iactrace.camera.detector.pmt.PMT`) is the detection response applied at
that surface.
"""

from .photodetector import ConstantQE, PhotoDetector, incidence_cos
from .pmt import PMT
from .surface import DetectionSurface

__all__ = [
    "DetectionSurface",
    "PhotoDetector",
    "ConstantQE",
    "PMT",
    "incidence_cos",
]
