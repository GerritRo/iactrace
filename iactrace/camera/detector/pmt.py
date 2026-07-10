from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...core.coatings import fresnel_unpolarized
from ...core.ray_bundle import RayBundle
from .photodetector import PhotoDetector, incidence_cos
from .surface import DetectionSurface


class PMT(PhotoDetector):
    """A photomultiplier: a photocathode (sensor) surface + a cylindrical body.

    A self-contained photodetector bundling everything a PMT contributes to
    detection, applied *after* the chain hands rays over to the photocathode:

    * **Geometry.** The **photocathode** is the sensor surface the chain traces
      rays onto (:attr:`surface`): a spherical cap of aperture ``face_radius``
      bulging ``face_sag`` toward the light (``face_sag = 0`` -> flat window).
      The non-photodetector **body** is a cylinder of the *same* radius extending
      ``length`` behind the photocathode rim, drawn by :meth:`envelope` for the
      diagnostic viz. Because the body sits entirely behind the rim (below the
      detector plane) at the photocathode radius, it fits the sensor surface
      with no intersection.
    * **Efficiency.** A single detection efficiency ``qe`` is applied to every
      ray landing on the photocathode. Real PMT efficiencies are measured with
      the entrance glass in place, so ``qe`` is the whole measured number and
      needs no separate window term -- this is the default (``n_window = None``).
    * **Optional entrance window.** Set ``n_window`` to also weight each ray by
      the unpolarized Fresnel transmittance at the air/window interface for its
      incident angle -- the angular response a single measured scalar cannot
      capture. When you do, ``qe`` should be the intrinsic photocathode QE (the
      glass loss is then modelled by the Fresnel term, not folded into ``qe``).

    A non-spherical photocathode figure can be modelled by subclassing and
    overriding :attr:`surface` with a
    :class:`~iactrace.camera.detector.surface.DetectionSurface` built from any core
    :class:`~iactrace.core.surfaces.SurfaceGroup`.

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

    def detect(self, local_rays: RayBundle) -> RayBundle:
        values = local_rays.values * self.qe
        if self.n_window is not None:
            # Incidence from the true surface normal at each landing position
            # (the PMT owns its photocathode geometry), so the Fresnel weight
            # is correct on the curved photocathode too.
            normals = self.surface.normals_at(local_rays.origins)
            cos_theta_i = incidence_cos(local_rays.directions, normals)
            _, T = fresnel_unpolarized(cos_theta_i, 1.0, self.n_window)
            values = values * T
        return local_rays.replace(values=values)

    @property
    def surface(self) -> DetectionSurface:
        """The photocathode surface -- the sensor front (a spherical cap).

        Aperture ``face_radius`` bulging ``face_sag`` toward the light (apex at
        ``+face_sag`` above the detector plane), so the traced photocathode
        matches the drawn body rim. ``face_sag = 0`` is a flat window.
        """
        a, h = self.face_radius, self.face_sag
        if h <= 0.0:
            return DetectionSurface(vertex_z=0.0, radius=a)
        r_curv = (a * a + h * h) / (2.0 * h)
        return DetectionSurface(vertex_z=h, curvature=-1.0 / r_curv, radius=a)

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
