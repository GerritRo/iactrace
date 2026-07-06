from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.coatings import fresnel_unpolarized
from ..core.intersections import intersect_conic_normal
from ..core.ray_bundle import RayBundle


def incidence_cos(directions: Array, normals: Array | None) -> Array:
    """Incidence cosine of each ray on a surface with the given outward normals.

    Mirrors the convention of :class:`~iactrace.core.interactions.RefractInteraction`
    (``cos_i = -dot(direction, normal)``). ``normals = None`` means a flat detector
    with normal ``+z``, i.e. ``-dir_z`` -- the legacy assumption -- so photosensors
    stay correct when called without geometry.
    """
    if normals is None:
        cos_i = -directions[:, 2]
    else:
        cos_i = -jnp.sum(directions * normals, axis=-1)
    return jnp.clip(cos_i, 0.0, 1.0)


class PhotoSensor(eqx.Module):
    """Abstract base for photosensor (PMT / SiPM) response models.

    A photosensor is the terminal response of a detection chain: it receives the
    rays the chain has traced onto its stopping surface -- together with the
    per-ray surface ``normals`` -- and weights ``values`` by its detection
    efficiency. Taking the normal alongside the ray (rather than a pre-rotated
    direction) matches the package's surface-interaction convention (see
    :class:`~iactrace.core.interactions.RefractInteraction`) and keeps the true
    ray direction available.
    """

    @abstractmethod
    def detect(self, local_rays: RayBundle, normals: Array | None = None) -> RayBundle:
        """Weight *local_rays* by detection efficiency at the stopping surface.

        Args:
            local_rays: Rays at the surface, pixel-local frame (true directions).
            normals: Per-ray outward surface normals ``(N, 3)`` for angle-dependent
                response; ``None`` assumes a flat detector (normal ``+z``). Use
                :func:`incidence_cos` to turn these into the incidence cosine.

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

    def envelope(self) -> tuple[Array, Array] | None:
        """Optional 3D envelope ``(z, rings)`` for :func:`iactrace.viz.show_sensor_chain`.

        Mirrors :meth:`~iactrace.camera.concentrator.Concentrator.cross_sections`
        for the detector side: a surface of revolution / lofted wall drawn around
        the detector plane so a physical photosensor body (e.g. a PMT's glass
        front + tube) becomes visible.

        * ``z``; shape ``(K,)`` axial samples in the pixel-local frame, with
          ``z = 0`` at the photocathode (detector) plane and ``+z`` toward the
          incoming light. The viz offsets these to
          :attr:`~iactrace.camera.chain.DetectionChain.detector_z`.
        * ``rings``; shape ``(K, M, 2)`` wall cross-section at each slice
          (large ``M`` ~ round).

        The default returns ``None`` ("no envelope drawn"); photosensors with a
        physical body override it.
        """
        return None

    def stopping_surface(self) -> StopSurface | None:
        """The geometric surface the chain traces rays onto for this photosensor.

        A photosensor *owns* its photocathode geometry: the chain reads this
        surface, traces rays to it, and hands them back (with normals) to
        :meth:`detect`. The default returns ``None`` -- a flat detector at the
        chain's detector plane -- so a photosensor without an explicit surface
        keeps the legacy behaviour. Photosensors carrying a curved / apertured
        photocathode return their :class:`StopSurface`.
        """
        return None


class ConstantQE(PhotoSensor):
    """Flat scalar quantum efficiency with no spatial or angular structure.

    The simplest photosensor and the default detector response: a single
    efficiency ``qe`` applied uniformly to every ray reaching the surface. Use
    it for a measured detection efficiency you want applied as a plain scalar,
    or as a perfect (``qe = 1``) pass-through.

    Args:
        qe: Quantum efficiency in ``[0, 1]``.
        surface: Optional photocathode geometry (a :class:`StopSurface`) the
            chain traces rays onto; ``None`` -> a flat detector at the chain's
            detector plane. A curved detector is
            ``StopSurface(curvature=..., radius=...)``.
    """

    qe: float = eqx.field(static=True)
    surface: StopSurface | None

    def __init__(self, qe: float = 1.0, surface: StopSurface | None = None) -> None:
        if not 0.0 <= qe <= 1.0:
            raise ValueError(f"qe must be in [0, 1], got {qe}")
        self.qe = float(qe)
        self.surface = surface

    def stopping_surface(self) -> StopSurface | None:
        return self.surface

    def detect(self, local_rays: RayBundle, normals: Array | None = None) -> RayBundle:
        return RayBundle(
            origins=local_rays.origins,
            directions=local_rays.directions,
            values=local_rays.values * self.qe,
            path_length=local_rays.path_length,
            n=local_rays.n,
            alive=local_rays.alive,
        )


class PMT(PhotoSensor):
    """A photomultiplier: a photocathode (sensor) surface + a cylindrical body.

    A self-contained photosensor bundling everything a PMT contributes to
    detection, applied *after* the chain hands rays over to the photocathode:

    * **Geometry.** The **photocathode** is the sensor surface the chain traces
      rays onto (:meth:`stopping_surface`): a spherical cap of aperture
      ``face_radius`` bulging ``face_sag`` toward the light (``face_sag = 0`` ->
      flat window). The non-photosensor **body** is a cylinder of the *same*
      radius extending ``length`` behind the photocathode rim, drawn by
      :meth:`envelope` for the diagnostic viz. Because the body sits entirely
      behind the rim (below the detector plane) at the photocathode radius, it
      fits the sensor surface with no intersection.
    * **Efficiency.** A single detection efficiency ``qe`` is applied to every
      ray landing on the photocathode. Real PMT efficiencies are measured with
      the entrance glass in place, so ``qe`` is the whole measured number and
      needs no separate window term -- this is the default (``n_window = None``).
    * **Optional entrance window.** Set ``n_window`` to also weight each ray by
      the unpolarized Fresnel transmittance at the air/window interface for its
      incident angle -- the angular response a single measured scalar cannot
      capture. When you do, ``qe`` should be the intrinsic photocathode QE (the
      glass loss is then modelled by the Fresnel term, not folded into ``qe``).

    Because Equinox modules are immutable pytrees, a **single** ``PMT`` instance
    is shared by reference across every pixel and every sensor whose group
    references it: build it once and hand the same object to each
    :class:`~iactrace.camera.sensor_group.SensorGroup`.

    Args:
        qe: Detection efficiency in ``[0, 1]`` applied at the photocathode.
            Defaults to ``1.0``.
        n_window: Refractive index of the entrance window. ``None`` (default)
            applies ``qe`` alone. A value ``> 1`` (e.g. ``1.48`` for borosilicate
            glass) additionally weights each ray by the incident-angle Fresnel
            transmittance through the window.
        face_radius: Photocathode aperture radius (and the body radius).
        face_sag: Bulge of the photocathode from rim to apex (``0`` = flat
            window). The front is a spherical cap whose radius of curvature is
            ``(face_radius**2 + face_sag**2) / (2 * face_sag)``.
        length: Axial length of the cylindrical body behind the photocathode.
            ``None`` defaults to ``2 * face_radius``.
        n_facets: Facets of the revolved body (``48`` ~ round).

    Raises:
        ValueError: on ``qe`` outside ``[0, 1]``, ``n_window <= 1``, non-positive
            ``face_radius``, negative ``face_sag`` / ``length``, or
            ``n_facets < 3``.
    """

    qe: float = eqx.field(static=True)
    n_window: float | None = eqx.field(static=True)
    face_radius: float = eqx.field(static=True)
    face_sag: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    n_facets: int = eqx.field(static=True)

    def __init__(
        self,
        qe: float = 1.0,
        *,
        n_window: float | None = None,
        face_radius: float,
        face_sag: float = 0.0,
        length: float | None = None,
        n_facets: int = 48,
    ) -> None:
        if not 0.0 <= qe <= 1.0:
            raise ValueError(f"qe must be in [0, 1], got {qe}")
        if n_window is not None and n_window <= 1.0:
            raise ValueError(f"n_window must be > 1.0, got {n_window}")
        if face_radius <= 0.0:
            raise ValueError(f"face_radius must be > 0, got {face_radius}")
        if face_sag < 0.0:
            raise ValueError(f"face_sag must be >= 0, got {face_sag}")
        if length is not None and length < 0.0:
            raise ValueError(f"length must be >= 0, got {length}")
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3, got {n_facets}")
        self.qe = float(qe)
        self.n_window = None if n_window is None else float(n_window)
        self.face_radius = float(face_radius)
        self.face_sag = float(face_sag)
        self.length = float(length) if length is not None else 2.0 * float(face_radius)
        self.n_facets = int(n_facets)

    def detect(self, local_rays: RayBundle, normals: Array | None = None) -> RayBundle:
        values = local_rays.values * self.qe
        if self.n_window is not None:
            # Incidence from the true surface normal (falls back to -dir_z for a
            # flat detector), so the Fresnel weight is correct on the curved
            # photocathode too.
            cos_theta_i = incidence_cos(local_rays.directions, normals)
            _, T = fresnel_unpolarized(cos_theta_i, 1.0, self.n_window)
            values = values * T
        return RayBundle(
            origins=local_rays.origins,
            directions=local_rays.directions,
            values=values,
            path_length=local_rays.path_length,
            n=local_rays.n,
            alive=local_rays.alive,
        )

    def stopping_surface(self) -> StopSurface:
        """The photocathode surface -- the sensor front (a spherical cap).

        Aperture ``face_radius`` bulging ``face_sag`` toward the light (apex at
        ``+face_sag`` above the detector plane), so the traced photocathode
        matches the drawn body rim. ``face_sag = 0`` is a flat window.
        """
        a, h = self.face_radius, self.face_sag
        if h <= 0.0:
            return StopSurface(vertex_z=0.0, radius=a)
        r_curv = (a * a + h * h) / (2.0 * h)
        return StopSurface(vertex_z=h, curvature=-1.0 / r_curv, radius=a)

    def outline(self) -> Array:
        ang = 2.0 * jnp.pi * jnp.arange(self.n_facets) / self.n_facets
        return self.face_radius * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)

    def envelope(self) -> tuple[Array, Array]:
        """Body-only cylinder for the viz: photocathode rim -> ``-length``.

        The photocathode (dome) front is the sensor surface, drawn separately; the
        body is just the tube behind it. Both share the rim circle (``z = 0``,
        ``r = face_radius``) and the body lies entirely at ``z <= 0``, so it fits
        the sensor surface with no intersection.
        """
        m = self.n_facets
        ang = 2.0 * jnp.pi * jnp.arange(m) / m
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # (M, 2)
        r_prof = jnp.array([self.face_radius, self.face_radius])
        z_prof = jnp.array([0.0, -self.length])
        rings = r_prof[:, None, None] * unit[None, :, :]  # (2, M, 2)
        return z_prof, rings


class StopSurface(eqx.Module):
    """The geometric surface a detection chain traces rays up to.

    Pure geometry, no detection physics. It is a conic of revolution with its
    vertex at ``z = vertex_z`` and aperture ``radius`` (``curvature = 0`` -> flat
    plane, ``curvature != 0`` with ``conic = 0`` -> sphere). The chain traces
    rays to this surface -- jointly with the cone walls when it can be hit
    mid-bounce, otherwise a straight advance -- and hands the resulting
    :class:`RayBundle` **plus the per-ray surface normals** to the
    :class:`PhotoSensor`, which owns quantum efficiency, window / Fresnel
    response, etc. The split is the point: geometry lives here, detection
    efficiencies live in the photosensor.

    Directions are kept in the true pixel-local frame; the surface normal travels
    alongside (as in :class:`~iactrace.core.interactions.RefractInteraction`), so
    the photosensor computes incidence as ``-dot(dir, normal)`` -- correct on a
    curved photocathode, and ``-dir_z`` for a flat one. ``values`` carry the
    throughput delivered to the surface (concentrator losses applied; ``0`` for
    rays that never reach it or land outside ``radius``).

    A surface owned by a :class:`PhotoSensor` places its ``vertex_z``
    **relative to the chain's detector plane** (``0`` = at the plane, ``+`` toward
    the light, i.e. peeking into a cone); the chain shifts it into absolute
    pixel-local coordinates via :meth:`shifted`. So the photocathode geometry is
    intrinsic to the photosensor and the ``gap`` handles mounting depth.

    Args:
        vertex_z: Axial position of the surface vertex, relative to the detector
            plane when owned by a photosensor (``0`` = at the plane).
        curvature: ``c = 1 / R``. ``0`` -> flat. ``> 0`` concave toward the
            incoming light (bowl); ``< 0`` convex (a dome bulging toward +z).
        conic: Conic constant ``k`` (``0`` -> sphere).
        radius: Aperture radius; rays landing beyond it are dropped. ``None`` ->
            unbounded.

    Raises:
        ValueError: on non-positive ``radius``.
    """

    vertex_z: float = eqx.field(static=True)
    curvature: float = eqx.field(static=True)
    conic: float = eqx.field(static=True)
    radius: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_z: float,
        curvature: float = 0.0,
        conic: float = 0.0,
        radius: float | None = None,
    ) -> None:
        if radius is not None and radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.vertex_z = float(vertex_z)
        self.curvature = float(curvature)
        self.conic = float(conic)
        self.radius = float("inf") if radius is None else float(radius)

    @property
    def is_flat(self) -> bool:
        return self.curvature == 0.0

    def shifted(self, dz: float) -> StopSurface:
        """Copy with ``vertex_z`` shifted by ``dz`` (relative -> absolute placement)."""
        return StopSurface(
            vertex_z=self.vertex_z + dz,
            curvature=self.curvature,
            conic=self.conic,
            radius=self.radius,
        )

    def needs_joint_trace(self, cone_length: float) -> bool:
        """Whether a cone of length ``cone_length`` must be traced jointly.

        True when the surface is curved or protrudes above the cone exit
        (``vertex_z > -cone_length``), so it can be hit mid-bounce; a flat
        surface at or below the exit is reached by a straight advance instead.
        """
        return (not self.is_flat) or (self.vertex_z > -cone_length + 1e-12)

    def _hit(self, o: Array, d: Array, vertex: float) -> tuple[Array, Array, Array, Array]:
        """Nearest forward hit for one ray, with the surface vertex at ``vertex``.

        ``vertex`` is the axial position of the surface in the *working* frame:
        ``vertex_z`` for the pixel-local no-cone trace, or ``vertex_z + length``
        in the cone frame. Returns ``(t, point, normal, within_aperture)``.
        """
        shift = jnp.array([0.0, 0.0, vertex])
        t, p_local, normal = intersect_conic_normal(o - shift, d, self.curvature, self.conic)
        point = p_local + shift
        within = (point[0] ** 2 + point[1] ** 2) <= self.radius**2
        return t, point, normal, within

    def intersect(self, rays: RayBundle) -> tuple[Array, Array, Array]:
        """Nearest forward hit of every ray on the surface (pixel-local frame).

        Returns ``(t, points, normals)`` -- ``inf`` ``t`` on a miss. The geometry
        hook the cone tracer uses alongside the cone walls.
        """
        t, pts, nrm, _ = jax.vmap(lambda o, d: self._hit(o, d, self.vertex_z))(
            rays.origins, rays.directions
        )
        return t, pts, nrm

    def stop(self, rays: RayBundle) -> tuple[RayBundle, Array]:
        """Advance *rays* onto the surface (no concentrator).

        A straight advance to the plane for a flat surface (handling rays already
        on it, ``t = 0``, and matching the legacy detector-plane drift), a conic
        intersection otherwise. Returns ``(rays_at_surface, normals)`` -- true
        directions preserved -- with ``values`` zeroed outside ``radius`` (or on a
        miss); the photosensor weights them using ``normals``.
        """
        o, d = rays.origins, rays.directions
        if self.is_flat:
            dz = d[:, 2]
            parallel = jnp.abs(dz) < 1e-12
            t = jnp.where(parallel, 0.0, (self.vertex_z - o[:, 2]) / jnp.where(parallel, 1.0, dz))
            point = o + t[:, None] * d
            within = (point[:, 0] ** 2 + point[:, 1] ** 2) <= self.radius**2
            normals = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), d.shape)
            out = RayBundle(
                origins=point,
                directions=d,
                values=jnp.where(within, rays.values, 0.0),
                path_length=rays.path_length + t * rays.n,
                n=rays.n,
                # Landing outside the photocathode aperture is a geometry loss.
                alive=rays.alive & within,
            )
            return out, normals
        t, point, normals, within = jax.vmap(lambda oi, di: self._hit(oi, di, self.vertex_z))(o, d)
        hit = jnp.isfinite(t) & (t > 0.0)
        ok = hit & within
        out = RayBundle(
            origins=jnp.where(ok[:, None], point, o),
            directions=d,
            values=jnp.where(ok, rays.values, 0.0),
            path_length=rays.path_length + jnp.where(hit, t, 0.0) * rays.n,
            n=rays.n,
            # Missing the surface or landing outside its aperture is geometry loss.
            alive=rays.alive & ok,
        )
        return out, normals
