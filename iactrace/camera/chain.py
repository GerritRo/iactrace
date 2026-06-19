from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..core.ray_bundle import RayBundle
from .concentrator import Concentrator
from .photosensor import PhotoSensor


class DetectionChain(eqx.Module):
    """A pixel's detection train: concentrator (optional) -> gap -> photosensor.

    The chain is identical for every pixel in a SensorGroup, so it runs once over
    all rays at once.

    Attributes:
        concentrator: Optional light concentrator (cone / lightguide).
        photosensor: Photosensor response model (always present).
        gap: Spacing from the upstream exit (cone exit, or the ``z = 0``
            entrance with no cone) to the detector plane. Defaults to ``0.0``.
    """

    concentrator: Concentrator | None
    photosensor: PhotoSensor
    gap: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if not self.gap >= 0.0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")

    @property
    def detector_z(self) -> float:
        """Axial position of the detector plane in the pixel-local frame.

        The single source of truth for where the train ends:
        ``-(concentrator length + gap)`` (or ``-gap`` with no concentrator).
        """
        length = self.concentrator.length if self.concentrator is not None else 0.0
        return -float(length) - self.gap

    def propagate(self, local_rays: RayBundle) -> RayBundle:
        """Run *local_rays* through the chain and return weighted rays.

        ``local_rays`` are in the pixel-local frame (entrance at ``z = 0``).
        The bundle is funnelled by the concentrator (if any), free-flighted
        across the ``gap`` to :attr:`detector_z`, then weighted by the
        photosensor. The returned bundle carries photoelectron-weighted
        ``values`` and the arrival ``path_length`` at the detector.

        Optical path length is tracked consistently across the train: the
        concentrator weights its internal leg by its own fill index (see
        :meth:`~iactrace.camera.concentrator.Concentrator.apply`), and the
        ``gap`` leg is weighted by the ray's medium index ``n`` — the medium the
        concentrator leaves the rays in (the camera body, usually air).
        """
        if self.concentrator is not None:
            local_rays = self.concentrator.apply(local_rays)
        if self.gap:
            local_rays = self._advance_to_detector(local_rays)
        return self.photosensor.detect(local_rays)

    def _advance_to_detector(self, rays: RayBundle) -> RayBundle:
        """Free-flight *rays* across the ``gap`` onto :attr:`detector_z`.

        The gap is the camera-body medium between the upstream exit and the
        detector; its optical path weights the geometric step by the ray's
        current refractive index ``n`` (air for a hollow cone, or whatever the
        concentrator reset ``n`` to at its exit aperture).
        """
        dz = rays.directions[:, 2]
        parallel = jnp.abs(dz) < 1e-10
        safe_dz = jnp.where(parallel, 1.0, dz)
        t = jnp.where(parallel, 0.0, (self.detector_z - rays.origins[:, 2]) / safe_dz)
        return RayBundle(
            origins=rays.origins + t[:, None] * rays.directions,
            directions=rays.directions,
            values=rays.values,
            path_length=rays.path_length + t * rays.n,  # OPL: weight by medium index
            n=rays.n,
        )
