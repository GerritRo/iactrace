from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ...core.ray_bundle import RayBundle
from ...core.trajectory import TraceResult
from ..detector.surface import DetectionSurface


class Concentrator(eqx.Module):
    """Abstract base for per-pixel light concentrators.

    A concentrator funnels light from its entrance aperture (``z = 0``) toward
    its exit aperture (``z = -length``) in its local space, onto a stopping
    surface. Its one transport primitive is :meth:`to_surface`: deliver rays
    from the entrance aperture onto a given
    :class:`~iactrace.camera.detector.surface.DetectionSurface`, tracing the
    concentrator's *own* internal geometry jointly with that surface.
    """

    length: eqx.AbstractVar[float]

    @abstractmethod
    def to_surface(self, rays: RayBundle, surface: DetectionSurface) -> RayBundle:
        """Deliver *rays* from the entrance aperture onto *surface*.

        Args:
            rays: Rays at the entrance aperture, pixel-local frame.
            surface: The stopping surface, placed in the pixel-local frame.

        Returns:
            Rays landed on *surface*, same frame.
        """
        raise NotImplementedError

    def trace_to_surface(self, rays: RayBundle, surface: DetectionSurface) -> TraceResult:
        """:meth:`to_surface`, additionally reporting the path rays took.

        Returns a :class:`~iactrace.core.trajectory.TraceResult` whose
        ``trajectory`` runs through the pixel-local frame, or is ``None`` when
        this concentrator cannot report a path -- the base implementation, which
        subclasses that trace internally (e.g.
        :class:`~iactrace.camera.optics.polygonal.PolygonalCone`) override.
        Callers fall back to a straight entrance-to-landing segment on ``None``.
        """
        return TraceResult(self.to_surface(rays, surface))

    def apply(self, rays: RayBundle) -> RayBundle:
        """Transport *rays* to the exit aperture (a flat plane at ``z = -length``).

        Convenience for standalone concentrator use / diagnostics:
        :meth:`to_surface` onto a flat, unbounded stop at the exit aperture.
        """
        return self.to_surface(rays, DetectionSurface(vertex_z=-self.length))

    def cross_sections(self) -> tuple[Array, Array] | None:
        """Optional geometry for :func:`iactrace.viz.show_sensor_chain`.

        Returns ``(z, rings)`` or ``None``:

        * ``z``; shape ``(K,)`` axial samples, ``z[0] = 0`` (entrance) ..
          ``z[-1] = -length`` (exit).
        * ``rings``; shape ``(K, M, 2)``: the ``M``-gon wall cross-section at
          each slice in the pixel-local frame (``M = 6`` hex, ``M = 4`` square,
          large ``M`` ~ round). ``rings[0]`` is the entrance aperture,
          ``rings[-1]`` the exit aperture.

        The default returns ``None`` ("not drawable"); concrete concentrators
        override it once they know their profile.
        """
        return None
