from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ...core.ray_bundle import RayBundle
from ..detector.surface import DetectionSurface


class Concentrator(eqx.Module):
    """Abstract base for per-pixel light concentrators (cones / light guides / lenses).

    A concentrator funnels light from its entrance aperture (``z = 0``) toward
    its exit aperture (``z = -length``) in its local space, onto a stopping
    surface. Its one transport primitive is :meth:`to_surface`: deliver rays
    from the entrance aperture onto a given
    :class:`~iactrace.camera.detector.surface.DetectionSurface`, tracing the
    concentrator's *own* internal geometry jointly with that surface.

    How it does so is entirely the concentrator's business, so the abstraction
    is agnostic to the physical mechanism: a hollow reflective cone bounces rays
    off its walls (:class:`~iactrace.camera.optics.polygonal.PolygonalCone`), a
    lens concentrator refracts them through its elements, a solid dielectric
    guide propagates and refracts at its faces. The detection chain only ever
    calls :meth:`to_surface` -- it never needs to know which kind of concentrator
    it holds, so a new design plugs in without touching the pipeline.
    """

    length: eqx.AbstractVar[float]

    @property
    def index(self) -> float:
        """Refractive index of the medium the concentrator is filled with.

        ``1.0`` for hollow / air-filled light guides such as Winston cones.
        Solid dielectric / lens concentrators override this; :meth:`to_surface`
        must weight the internal geometric path by it when accumulating optical
        path length (``OPL += index * geometric_length``).
        """
        return 1.0

    @abstractmethod
    def to_surface(self, rays: RayBundle, surface: DetectionSurface) -> RayBundle:
        """Deliver *rays* from the entrance aperture onto *surface*.

        The single transport primitive every concentrator implements, tracing
        its internal geometry jointly with the stopping surface (so a
        photocathode curved into, or set below, the concentrator is landed on
        correctly). Rays enter in the ``z = 0`` plane travelling toward ``-z``;
        the returned bundle sits on *surface* with true directions preserved,
        reflection / transmission losses folded into ``values`` and the optical
        path added to ``path_length``. Rays that never reach the surface (lost
        back through the entrance, absorbed, outside the mouth) come back with
        ``alive = False`` and ``values = 0``; their positions are meaningless, as
        everywhere else in the package.

        Args:
            rays: Rays at the entrance aperture, pixel-local frame.
            surface: The stopping surface, placed in the pixel-local frame.

        Returns:
            Rays landed on *surface*, same frame.
        """
        raise NotImplementedError

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
