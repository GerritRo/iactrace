from __future__ import annotations

import equinox as eqx

from ..core.ray_bundle import RayBundle
from .detector import DetectionSurface, PhotoDetector
from .optics import Concentrator


class DetectionChain(eqx.Module):
    """A pixel's detection train: (optional concentrator) -> surface -> photodetector.

    Every chain traces rays up to the photodetector's own sensor surface (its
    photocathode geometry, :attr:`~iactrace.camera.detector.photodetector.PhotoDetector.surface`)
    and hands the resulting bundle back to the photodetector, which applies its
    detection efficiencies (QE, window response, ...). Geometry is owned by the
    photodetector; the chain only *places* it, at the detector plane
    :attr:`detector_z` set by the concentrator + ``gap``. The chain is identical
    for every pixel in a SensorGroup, so it runs once over all rays at once.

    Attributes:
        concentrator: Optional light concentrator (cone / lightguide).
        photodetector: Photodetector -- both the response and (via its
            ``surface``) the photocathode geometry rays are traced to.
        gap: Spacing from the concentrator exit (or the entrance with no cone) to
            the detector plane where the photocathode is mounted. Defaults ``0.0``.
    """

    concentrator: Concentrator | None
    photodetector: PhotoDetector
    gap: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if not self.gap >= 0.0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")

    def with_concentrator(self, concentrator: Concentrator | None) -> DetectionChain:
        """Return a copy of this chain with its concentrator replaced."""
        return DetectionChain(concentrator, self.photodetector, self.gap)

    def with_photodetector(self, photodetector: PhotoDetector) -> DetectionChain:
        """Return a copy of this chain with its photodetector replaced."""
        return DetectionChain(self.concentrator, photodetector, self.gap)

    def with_gap(self, gap: float) -> DetectionChain:
        """Return a copy of this chain with its gap replaced."""
        return DetectionChain(self.concentrator, self.photodetector, float(gap))

    @property
    def detector_z(self) -> float:
        """Detector-plane position in the pixel-local frame: ``-(length + gap)``."""
        length = self.concentrator.length if self.concentrator is not None else 0.0
        return -float(length) - self.gap

    @property
    def surface(self) -> DetectionSurface:
        """The photodetector's sensor surface, placed at :attr:`detector_z`.

        The photodetector owns the surface with ``vertex_z`` relative to the
        detector plane; this property shifts it into absolute pixel-local
        coordinates -- the surface rays are actually traced onto. Public so
        diagnostics (e.g. :func:`iactrace.viz.show_sensor_chain`) can read
        the placed geometry.
        """
        return self.photodetector.surface.shifted(self.detector_z)

    def propagate(self, local_rays: RayBundle) -> RayBundle:
        """Trace *local_rays* to the sensor surface, then hand off to the photodetector.

        ``local_rays`` are in the pixel-local frame (entrance at ``z = 0``). With
        a concentrator, they are delivered to :attr:`surface` by the
        concentrator's own
        :meth:`~iactrace.camera.optics.concentrator.Concentrator.to_surface` -- the chain
        stays agnostic to how (a wall cone co-traces its walls with the surface so
        a curved or protruding photocathode is hit mid-bounce; a lens concentrator
        refracts then lands). With no concentrator the rays advance straight onto
        the surface. The handover to the photodetector is just the resulting bundle
        -- rays at the surface, pixel-local frame -- which it weights by its own
        detection efficiency (reading any geometry it needs from the surface it
        owns). Optical path length is accumulated up to the surface (concentrator
        fill index on its internal leg, ray medium ``n`` on the free legs).
        """
        surface = self.surface
        if self.concentrator is None:
            landed = surface.stop(local_rays)
        else:
            landed = self.concentrator.to_surface(local_rays, surface)
        return self.photodetector.detect(landed)
