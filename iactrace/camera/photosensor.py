from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ..core.ray_bundle import RayBundle


class PhotoSensor(eqx.Module):
    """Abstract base for photosensor (PMT / SiPM) response models.

    A photosensor operates in the **canonical pixel-local frame** defined in
    :mod:`iactrace.camera.chain` (light travels toward ``-z``).

    Rays arrive already at the detector plane.
    """

    @abstractmethod
    def detect(self, local_rays: RayBundle) -> RayBundle:
        """Weight *local_rays* by detection efficiency at the detector plane.

        Rays already lie in the detector plane (``z = chain.detector_z``).
        Implementations scale ``values`` by the (optionally position/angle-
        dependent) detection efficiency.

        Args:
            local_rays: Rays at the detector plane, pixel-local frame.

        Returns:
            Rays with photoelectron-weighted ``values``; geometry unchanged.
        """
        raise NotImplementedError

    def outline(self) -> Array | None:
        """Optional active-area polygon ``(M, 2)`` for the diagnostic viz.

        Expressed in the pixel-local frame. The default returns ``None``
        ("not drawable"), in which case :func:`iactrace.viz.show_sensor_chain`
        falls back to the entrance-aperture footprint.
        """
        return None


class UniformQE(PhotoSensor):
    """Flat scalar quantum efficiency with no spatial structure.

    The simplest photosensor: a single efficiency ``qe`` applied uniformly at
    the detector plane.
    """

    qe: float = eqx.field(static=True)

    def __init__(self, qe: float = 1.0) -> None:
        """Flat photosensor with quantum efficiency ``qe`` in ``[0, 1]``.

        Raises:
            ValueError: if ``qe`` is outside ``[0, 1]``.
        """
        if not 0.0 <= qe <= 1.0:
            raise ValueError(f"qe must be in [0, 1], got {qe}")
        self.qe = float(qe)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        return RayBundle(
            origins=local_rays.origins,
            directions=local_rays.directions,
            values=local_rays.values * self.qe,
            path_length=local_rays.path_length,
            n=local_rays.n,
        )
