from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...core.ray_bundle import RayBundle
from ...core.refractive_index import ConstantIndex, RefractiveIndex, as_refractive_index
from ...core.responses import ResponseCurve, fresnel_unpolarized
from ...core.surfaces import AsphericSurfaceGroup, SurfaceGroup
from .landing import AngularResponse
from .photodetector import PhotoDetector, _validate_qe, apply_qe, incidence_cos
from .surface import DetectionSurface


class PMT(PhotoDetector):
    """
    A sensor surface plus a cylindrical body.

    Each ray the chain traces onto the sensor surface is weighted by

        qe * qe_curve * angular_response * T_window

    where any factor not given is 1.

    Without window_index, the efficiency should be the measured datasheet
    value, which already includes the entrance glass. With it, T_window is
    the unpolarized air/window Fresnel transmittance at the ray's incidence
    angle, so the efficiency should be that of the photocathode.

    Parameters
    ----------
    qe
        Bulk efficiency scalar in [0, 1]. Default 1.0.
    qe_curve
        QE(lambda) as a ResponseCurve, optionally with an incidence-angle
        axis. None (default) is flat.
    angular_response
        Relative response against incidence angle, as an AngularResponse.
        May also depend on landing radius, direction azimuth and
        wavelength. None (default) is 1.
    window_index
        Entrance-window refractive index: a number > 1 (e.g. 1.48 for
        borosilicate) or a RefractiveIndex model for a dispersive
        n(lambda). None (default) applies no Fresnel term.
    face_radius
        Radius of the sensor aperture and of the body.
    surface
        Sensor surface figure: a single-element SurfaceGroup (typically an
        AsphericSurfaceGroup, optionally summed with a ZernikeSurfaceGroup
        via SumSurfaceGroup). None (default) is flat.
    vertex_z
        Axial position of the surface vertex relative to the detector
        plane: 0 is flush with the mount, > 0 protrudes toward the light
        (e.g. a domed window).
    length
        Axial length of the body behind the sensor surface. None (default)
        is 2 * face_radius.
    n_facets
        Facets of the revolved body (48 is approximately round).

    Raises
    ------
    ValueError
        If qe is outside [0, 1], a constant window_index is <= 1,
        face_radius is not positive, length is negative, or n_facets < 3.
    """

    qe: float = eqx.field(static=True)
    qe_curve: ResponseCurve | None
    angular_response: AngularResponse | None
    window_index: RefractiveIndex | None
    face_radius: float = eqx.field(static=True)
    shape: SurfaceGroup
    vertex_z: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    n_facets: int = eqx.field(static=True)

    def __init__(
        self,
        qe: float = 1.0,
        *,
        qe_curve: ResponseCurve | None = None,
        angular_response: AngularResponse | None = None,
        window_index: RefractiveIndex | Array | float | None = None,
        face_radius: float,
        surface: SurfaceGroup | None = None,
        vertex_z: float = 0.0,
        length: float | None = None,
        n_facets: int = 48,
    ) -> None:
        if face_radius <= 0.0:
            raise ValueError(f"face_radius must be > 0, got {face_radius}")
        if length is not None and length < 0.0:
            raise ValueError(f"length must be >= 0, got {length}")
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3, got {n_facets}")
        self.qe = _validate_qe(qe)
        self.qe_curve = qe_curve
        self.angular_response = angular_response
        self.window_index = None if window_index is None else as_refractive_index(window_index, 1)
        if (
            isinstance(self.window_index, ConstantIndex)
            and float(self.window_index.values[0]) <= 1.0
        ):
            raise ValueError(f"window_index must be > 1.0, got {window_index}")
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
        normals = self.surface.normals_at(local_rays.origins)
        cos_theta_i = incidence_cos(local_rays.directions, normals)
        values = apply_qe(local_rays, self.qe, self.qe_curve, cos_theta_i)
        elem = jnp.zeros(local_rays.wavelength.shape[0], dtype=jnp.int32)
        if self.angular_response is not None:
            values = values * self.angular_response.at(
                local_rays.origins, local_rays.directions, normals, local_rays.wavelength
            )
        if self.window_index is not None:
            n_window = self.window_index.n_at(elem, local_rays.wavelength)
            _, T = fresnel_unpolarized(cos_theta_i, 1.0, n_window)
            values = values * T
        return local_rays.replace(values=values)

    @property
    def surface(self) -> DetectionSurface:
        """The sensor surface.

        shape supplies the figure (flat by default; curved / aspheric /
        Zernike otherwise), placed with its vertex at vertex_z and
        bounded by face_radius.
        """
        return DetectionSurface(self.shape, vertex_z=self.vertex_z, radius=self.face_radius)

    def outline(self) -> Array:
        ang = 2.0 * jnp.pi * jnp.arange(self.n_facets) / self.n_facets
        return self.face_radius * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)

    def envelope(self) -> tuple[Array, Array]:
        """Body-only cylinder for the viz: vertex_z -> vertex_z - length."""
        m = self.n_facets
        ang = 2.0 * jnp.pi * jnp.arange(m) / m
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)
        r_prof = jnp.array([self.face_radius, self.face_radius])
        rim_z = self.vertex_z + self.shape.sag_at(0, self.face_radius, 0.0)
        z_prof = jnp.array([rim_z, rim_z - self.length])
        rings = r_prof[:, None, None] * unit[None, :, :]
        return z_prof, rings
