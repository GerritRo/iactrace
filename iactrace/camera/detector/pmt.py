from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...core.coatings import fresnel_unpolarized
from ...core.ray_bundle import RayBundle
from ...core.surfaces import AsphericSurfaceGroup, SurfaceGroup
from .photodetector import PhotoDetector, _validate_qe, incidence_cos
from .surface import DetectionSurface


class PMT(PhotoDetector):
    """A photomultiplier: a sensor surface + a cylindrical body.

    A self-contained photodetector bundling everything a PMT contributes to
    detection, applied *after* the chain hands rays over to the sensor surface:

    * **Geometry.** The **sensor surface** is the sensor surface the chain traces
      rays onto (:attr:`surface`): bounded by ``face_radius`` and placed with
      its vertex at ``vertex_z`` (relative to the detector plane; ``0`` = flush
      with the mount, ``> 0`` peeks toward the light, as for a domed window).
    * **Efficiency.** A single detection efficiency ``qe`` is applied to every
      ray landing on the sensor surface. Real PMT efficiencies are measured with
      the entrance glass in place, so ``qe`` is the whole measured number and
      needs no separate window term -- this is the default (``n_window = None``).
    * **Optional entrance window.** Set ``n_window`` to also weight each ray by
      the unpolarized Fresnel transmittance at the air/window interface for its
      incident angle -- the angular response a single measured scalar cannot
      capture. When you do, ``qe`` should be the intrinsic photocathode QE (the
      glass loss is then modelled by the Fresnel term, not folded into ``qe``).

    A :class:`~iactrace.core.surfaces.FreeformSurfaceGroup` sensor is
    supported at the Python level (pass it as ``surface``), but -- like a
    freeform mirror or lens surface -- is not representable in YAML.

    Args:
        qe: Detection efficiency in ``[0, 1]`` applied at the photocathode.
            Defaults to ``1.0``.
        n_window: Refractive index of the entrance window. ``None`` (default)
            applies ``qe`` alone. A value ``> 1`` (e.g. ``1.48`` for borosilicate
            glass) additionally weights each ray by the incident-angle Fresnel
            transmittance through the window.
        face_radius: Sensor surface aperture radius (and the body radius).
        surface: The sensor surface figure, a single-element
            :class:`~iactrace.core.surfaces.SurfaceGroup` (typically an
            :class:`~iactrace.core.surfaces.AsphericSurfaceGroup`, optionally
            summed with a :class:`~iactrace.core.surfaces.ZernikeSurfaceGroup`
            via :class:`~iactrace.core.surfaces.SumSurfaceGroup`). ``None``
            (default) is a flat window.
        vertex_z: Axial position of the surface's vertex, relative to the
            detector plane (``0`` = at the plane; same convention as
            :attr:`~iactrace.camera.detector.surface.DetectionSurface.vertex_z`).
        length: Axial length of the cylindrical body behind the sensor surface.
            ``None`` defaults to ``2 * face_radius``.
        n_facets: Facets of the revolved body (``48`` ~ round).

    Raises:
        ValueError: on ``qe`` outside ``[0, 1]``, ``n_window <= 1``, non-positive
            ``face_radius``, negative ``length``, or ``n_facets < 3``.
    """

    qe: float = eqx.field(static=True)
    n_window: float | None = eqx.field(static=True)
    face_radius: float = eqx.field(static=True)
    shape: SurfaceGroup
    vertex_z: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    n_facets: int = eqx.field(static=True)

    def __init__(
        self,
        qe: float = 1.0,
        *,
        n_window: float | None = None,
        face_radius: float,
        surface: SurfaceGroup | None = None,
        vertex_z: float = 0.0,
        length: float | None = None,
        n_facets: int = 48,
    ) -> None:
        if n_window is not None and n_window <= 1.0:
            raise ValueError(f"n_window must be > 1.0, got {n_window}")
        if face_radius <= 0.0:
            raise ValueError(f"face_radius must be > 0, got {face_radius}")
        if length is not None and length < 0.0:
            raise ValueError(f"length must be >= 0, got {length}")
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3, got {n_facets}")
        self.qe = _validate_qe(qe)
        self.n_window = None if n_window is None else float(n_window)
        self.face_radius = float(face_radius)
        self.shape = (
            surface
            if surface is not None
            else AsphericSurfaceGroup(
                offsets=jnp.zeros((1, 2)),
                curvatures=jnp.zeros(1),
                conics=jnp.zeros(1),
                aspherics=jnp.zeros((1, 0)),
            )
        )
        self.vertex_z = float(vertex_z)
        self.length = float(length) if length is not None else 2.0 * float(face_radius)
        self.n_facets = int(n_facets)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        values = local_rays.values * self.qe
        if self.n_window is not None:
            # Incidence from the true surface normal at each landing position
            # (the PMT owns its sensor surface geometry), so the Fresnel weight
            # is correct on the curved sensor surface too.
            normals = self.surface.normals_at(local_rays.origins)
            cos_theta_i = incidence_cos(local_rays.directions, normals)
            _, T = fresnel_unpolarized(cos_theta_i, 1.0, self.n_window)
            values = values * T
        return local_rays.replace(values=values)

    @property
    def surface(self) -> DetectionSurface:
        """The sensor surface.

        :attr:`shape` supplies the figure (flat by default; curved / aspheric /
        Zernike otherwise), placed with its vertex at :attr:`vertex_z` and
        bounded by :attr:`face_radius` -- the exact same
        :class:`~iactrace.camera.detector.surface.DetectionSurface` machinery
        used by every other photodetector's surface.
        """
        return DetectionSurface(self.shape, vertex_z=self.vertex_z, radius=self.face_radius)

    def outline(self) -> Array:
        ang = 2.0 * jnp.pi * jnp.arange(self.n_facets) / self.n_facets
        return self.face_radius * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)

    def envelope(self) -> tuple[Array, Array]:
        """Body-only cylinder for the viz: :attr:`vertex_z` -> ``vertex_z - length``.

        The entry window is the sensor surface, drawn separately; the
        body is just the tube behind it, sharing the rim circle
        (``z = vertex_z``, ``r = face_radius``) with a flat / recessed
        sensor surface so the two fit without intersection. A sensor surface that
        bulges *past* its rim (a strongly domed :attr:`shape`) is drawn with
        its body starting at the mount rather than the true apex; this is a
        diagnostic-viz simplification only, not a tracing concern.
        """
        m = self.n_facets
        ang = 2.0 * jnp.pi * jnp.arange(m) / m
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # (M, 2)
        r_prof = jnp.array([self.face_radius, self.face_radius])
        z_prof = jnp.array([self.vertex_z, self.vertex_z - self.length])
        rings = r_prof[:, None, None] * unit[None, :, :]  # (2, M, 2)
        return z_prof, rings
