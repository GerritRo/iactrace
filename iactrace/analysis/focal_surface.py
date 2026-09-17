from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.intersections import (
    intersect_conic,
    intersect_plane,
    is_hit,
    newton_raphson_intersect,
)
from ..core.ray_bundle import RayBundle
from ..core.render import LazyRayBundle
from ..core.surfaces import sag_raw
from ..core.transforms import euler_to_matrix


class FocalSurfaceHits(eqx.Module):
    """Result of intersecting a RayBundle with a FocalSurface.

    Attributes
    ----------
    xy_local
        Tangent-plane coordinates at the hit, in the surface-local
        frame (n_rays, 2).
    z_local
        Sag of the surface at the hit (n_rays,). Always 0 for a flat
        focal plane.
    t
        Ray parameter at the hit; equals the world-frame distance travelled
        from ray_bundle.origins to the surface (n_rays,).
    hit_mask
        Liveness at the surface (n_rays,).
    directions_local : array, shape (n_rays, 3)
        Ray directions in the surface-local frame.
    opl
        Per-ray optical path length from the source wavefront to the
        focal surface, in metres (n_rays,). Equals
        ray_bundle.path_length + t * ray_bundle.n.
    values
        Throughput-weighted intensity carried to the surface (n_rays,),
        in the input bundle's units.
    wavelength : array, shape (n_rays,)
        Per-ray wavelength carried from the input bundle.
    """

    xy_local: Array
    z_local: Array
    t: Array
    hit_mask: Array
    directions_local: Array
    opl: Array
    values: Array
    wavelength: Array

    @property
    def alive(self) -> Array:
        """Rays that reached the surface carrying light (hit_mask & values > 0)."""
        return self.hit_mask & (self.values > 0)


class FocalSurface(eqx.Module):
    """Abstract focal surface attached to the camera frame."""

    position: Array  # (3,) vertex of the surface in the camera/telescope frame
    rotation: Array  # (3,) Euler XYZ degrees; local +Z is the normal at the vertex

    @abstractmethod
    def _intersect_local(
        self,
        o_local: Array,
        d_local: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Intersect a single local-frame ray with the surface.

        Parameters
        ----------
        o_local : array, shape (3,)
            Ray origin in the surface-local frame.
        d_local
            Ray direction in the surface-local frame (3,), normalized.

        Returns
        -------
        Tuple (t, xy_local, z_local, valid) where:
            - t is the ray parameter (scalar),
            - xy_local is the in-plane hit coordinate (2,),
            - z_local is the surface sag at the hit (scalar),
            - valid is a bool flag indicating a real intersection.
        """
        raise NotImplementedError

    def intersect(
        self,
        ray_bundle: RayBundle | LazyRayBundle,
    ) -> FocalSurfaceHits:
        """Intersect every ray in ray_bundle with this focal surface.

        Lazy bundles are materialised first; per-ray analysis cannot fold.
        """
        if isinstance(ray_bundle, LazyRayBundle):
            ray_bundle = ray_bundle.materialise()

        rot = euler_to_matrix(self.rotation)

        o_local = (ray_bundle.origins - self.position) @ rot
        d_local = ray_bundle.directions @ rot

        t, xy_local, z_local, valid = jax.vmap(self._intersect_local)(
            o_local,
            d_local,
        )
        # Liveness at the surface: only rays that were still alive coming in
        # and land on a real intersection.
        hit_mask = ray_bundle.alive & valid & jnp.isfinite(t)

        safe_t = jnp.where(hit_mask, t, 0.0)
        opl = ray_bundle.path_length + safe_t * ray_bundle.n

        return FocalSurfaceHits(
            xy_local=xy_local,
            z_local=z_local,
            t=t,
            hit_mask=hit_mask,
            directions_local=d_local,
            opl=opl,
            values=ray_bundle.values,
            wavelength=ray_bundle.wavelength,
        )


class FlatFocalPlane(FocalSurface):
    """A flat focal plane."""

    def __init__(
        self,
        position: Array | None = None,
        rotation: Array | None = None,
    ) -> None:
        self.position = jnp.zeros(3) if position is None else jnp.asarray(position, dtype=float)
        self.rotation = jnp.zeros(3) if rotation is None else jnp.asarray(rotation, dtype=float)

    def _intersect_local(self, o_local, d_local):
        center = jnp.zeros(3)
        rot = jnp.eye(3)
        xy, t = intersect_plane(o_local, d_local, center, rot)
        valid = is_hit(t)
        z = jnp.zeros((), dtype=xy.dtype)
        return t, xy, z, valid


class AsphericFocalSurface(FocalSurface):
    """A rotationally-symmetric aspheric focal surface.

    Parameterisation matches AsphericSurfaceGroup; the surface sag is

        z(r) = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2))
               + sum_i a_i r^(2i+4)
    """

    curvature: Array
    conic: Array
    aspherics: Array

    def __init__(
        self,
        position: Array | None = None,
        rotation: Array | None = None,
        *,
        curvature: float | Array = 0.0,
        conic: float | Array = 0.0,
        aspherics: Array | None = None,
    ) -> None:
        self.position = jnp.zeros(3) if position is None else jnp.asarray(position, dtype=float)
        self.rotation = jnp.zeros(3) if rotation is None else jnp.asarray(rotation, dtype=float)
        self.curvature = jnp.asarray(curvature, dtype=float)
        self.conic = jnp.asarray(conic, dtype=float)
        self.aspherics = (
            jnp.zeros((0,)) if aspherics is None else jnp.asarray(aspherics, dtype=float)
        )

    def _intersect_local(self, o_local, d_local):
        c = self.curvature
        k = self.conic
        a = self.aspherics

        t_init = intersect_conic(o_local, d_local, c, k)
        t, hit_xy, valid = newton_raphson_intersect(
            lambda x, y: sag_raw(x, y, c, k, a),
            o_local,
            d_local,
            t_init,
        )
        z = sag_raw(hit_xy[0], hit_xy[1], c, k, a)
        return t, hit_xy, z, valid
