from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ...core.ray_bundle import RayBundle
from ...core.trajectory import TraceResult
from ..detector.surface import DetectionSurface


class Concentrator(eqx.Module):
    """Abstract base for per-pixel light concentrators."""

    length: eqx.AbstractVar[float]

    @abstractmethod
    def to_surface(self, rays: RayBundle, surface: DetectionSurface) -> RayBundle:
        """Deliver rays from the entrance aperture onto surface.

        Parameters
        ----------
        rays
            Rays at the entrance aperture, pixel-local frame.
        surface
            The stopping surface, placed in the pixel-local frame.

        Returns
        -------
        Rays landed on surface, same frame.
        """
        raise NotImplementedError

    def trace_to_surface(self, rays: RayBundle, surface: DetectionSurface) -> TraceResult:
        """Same as to_surface, additionally reporting the path rays took."""
        return TraceResult(self.to_surface(rays, surface))

    def apply(self, rays: RayBundle) -> RayBundle:
        """Transport rays to the exit aperture (a flat plane at z = -length)."""
        return self.to_surface(rays, DetectionSurface(vertex_z=-self.length))

    def cross_sections(self) -> tuple[Array, Array] | None:
        """Optional geometry for iactrace.viz.show_sensor_chain."""
        return None
