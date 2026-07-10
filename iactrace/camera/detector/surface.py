from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ...core.ray_bundle import RayBundle
from ...core.surfaces import AsphericSurfaceGroup, SurfaceGroup


class DetectionSurface(eqx.Module):
    """The sensor surface a detection chain delivers rays onto.

    Pure geometry, no detection physics; every :class:`PhotoDetector` owns one
    (:attr:`PhotoDetector.surface`). It is a surface of revolution with its
    vertex at ``z = vertex_z`` and aperture ``radius``, described either by

    * ``curvature`` / ``conic`` -- a plane (``curvature = 0``) or conic of
      revolution (a pure-conic core surface, intersected in closed form), or
    * ``shape`` -- any single-element core
      :class:`~iactrace.core.surfaces.SurfaceGroup` (aspheric, Zernike,
      freeform, sums). This is the escape hatch for complicated photocathode
      figures; it shares the exact surface conventions and intersection
      machinery of the optical elements (Newton-Raphson, bypassed for pure
      conics).

    The chain traces rays to this surface -- jointly with the concentrator
    walls when the pixel has a wall-based concentrator, otherwise a straight
    advance -- and hands the resulting :class:`RayBundle` to the
    :class:`PhotoDetector`, which owns quantum efficiency, window / Fresnel
    response, etc. The split is the point: geometry lives here, detection
    efficiencies live in the photodetector (which reads any geometry it needs,
    e.g. :meth:`normals_at`, from the surface it owns).

    ``vertex_z`` is **relative to the chain's detector plane** (``0`` = at the
    plane, ``+`` toward the light, i.e. peeking into a cone); the chain shifts
    it into absolute pixel-local coordinates via :meth:`shifted`. So the
    photocathode geometry is intrinsic to the photodetector and the chain's
    ``gap`` handles mounting depth.

    Args:
        shape: Optional single-element core surface group giving the sag
            ``z(x, y)`` in the vertex frame (element ``0`` is used). Mutually
            exclusive with ``curvature`` / ``conic``.
        vertex_z: Axial position of the surface vertex, relative to the
            detector plane (``0`` = at the plane).
        curvature: ``c = 1 / R``. ``0`` -> flat. ``> 0`` concave toward the
            incoming light (bowl); ``< 0`` convex (a dome bulging toward +z).
        conic: Conic constant ``k`` (``0`` -> sphere).
        radius: Aperture radius; rays landing beyond it are dropped. ``None``
            -> unbounded.

    Raises:
        ValueError: on non-positive ``radius``, or when both ``shape`` and
            ``curvature`` / ``conic`` are given.
    """

    shape: SurfaceGroup
    vertex_z: float = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    is_flat: bool = eqx.field(static=True)

    def __init__(
        self,
        shape: SurfaceGroup | None = None,
        *,
        vertex_z: float = 0.0,
        curvature: float = 0.0,
        conic: float = 0.0,
        radius: float | None = None,
    ) -> None:
        if radius is not None and radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if shape is not None and (curvature != 0.0 or conic != 0.0):
            raise ValueError("give either a core surface `shape` or curvature/conic, not both")
        self.vertex_z = float(vertex_z)
        self.radius = float("inf") if radius is None else float(radius)
        self.is_flat = shape is None and curvature == 0.0
        if shape is None:
            # A pure-conic core surface: no aspheric terms, so the shared
            # intersection machinery takes the closed-form root (no Newton).
            self.shape = AsphericSurfaceGroup(
                offsets=jnp.zeros((1, 2)),
                curvatures=jnp.asarray([float(curvature)]),
                conics=jnp.asarray([float(conic)]),
                aspherics=jnp.zeros((1, 0)),
            )
        else:
            self.shape = shape

    def shifted(self, dz: float) -> DetectionSurface:
        """Copy with ``vertex_z`` shifted by ``dz`` (relative -> absolute placement)."""
        if self.is_flat:
            return DetectionSurface(vertex_z=self.vertex_z + dz, radius=self.radius)
        return DetectionSurface(self.shape, vertex_z=self.vertex_z + dz, radius=self.radius)

    def _hit(self, o: Array, d: Array, vertex: float) -> tuple[Array, Array, Array]:
        """Nearest forward hit for one ray, with the surface vertex at ``vertex``.

        ``vertex`` is the axial position of the surface in the *working* frame:
        ``vertex_z`` for the pixel-local no-cone trace, or ``vertex_z + length``
        in the cone frame. Resolved by the core surface machinery
        (:meth:`~iactrace.core.surfaces.SurfaceGroup._intersect_t`). Returns
        ``(t, point, within_aperture)`` with ``t = inf`` (and a finite,
        sanitized ``point``) on a miss.
        """
        shift = jnp.array([0.0, 0.0, vertex])
        t = self.shape._index(0)._intersect_t(o - shift, d)
        point = o + jnp.where(jnp.isfinite(t), t, 0.0) * d
        within = (point[0] ** 2 + point[1] ** 2) <= self.radius**2
        return t, point, within

    def normals_at(self, points: Array) -> Array:
        """Outward unit surface normals at the transverse positions of *points*.

        The surface is ``z = vertex_z + sag(x, y)``, so only ``(x, y)`` matter
        and the result is placement-independent -- angle-dependent photodetectors
        call this on the landing ``origins`` handed over by the chain.
        Positions are clamped into the aperture first, so the meaningless
        positions of dead rays cannot push the sag out of its domain (their
        weight is already ``0``; the normal must merely stay finite).
        """
        x, y = points[..., 0], points[..., 1]
        if math.isfinite(self.radius):
            r2 = x**2 + y**2
            over = r2 > self.radius**2
            scale = jnp.where(over, self.radius / jnp.sqrt(jnp.where(over, r2, 1.0)), 1.0)
            x, y = x * scale, y * scale
        elem = self.shape._index(0)
        _, normals = jax.vmap(elem.compute_sag_and_normal_at)(x, y)
        return normals

    def stop(self, rays: RayBundle) -> RayBundle:
        """Advance *rays* onto the surface (no concentrator).

        A straight advance to the plane for a flat surface (handling rays already
        on it, ``t = 0``, and matching the legacy detector-plane drift), a surface
        intersection otherwise. Returns the rays at the surface -- true directions
        preserved -- with ``values`` zeroed outside ``radius`` (or on a miss).
        """
        o, d = rays.origins, rays.directions
        if self.is_flat:
            dz = d[:, 2]
            parallel = jnp.abs(dz) < 1e-12
            t = jnp.where(parallel, 0.0, (self.vertex_z - o[:, 2]) / jnp.where(parallel, 1.0, dz))
            point = o + t[:, None] * d
            within = (point[:, 0] ** 2 + point[:, 1] ** 2) <= self.radius**2
            return rays.replace(
                origins=point,
                values=jnp.where(within, rays.values, 0.0),
                path_length=rays.path_length + t * rays.n,
                # Landing outside the photocathode aperture is a geometry loss.
                alive=rays.alive & within,
            )
        t, point, within = jax.vmap(lambda oi, di: self._hit(oi, di, self.vertex_z))(o, d)
        hit = jnp.isfinite(t) & (t > 0.0)
        ok = hit & within
        return rays.replace(
            origins=jnp.where(ok[:, None], point, o),
            values=jnp.where(ok, rays.values, 0.0),
            path_length=rays.path_length + jnp.where(hit, t, 0.0) * rays.n,
            # Missing the surface or landing outside its aperture is geometry loss.
            alive=rays.alive & ok,
        )
