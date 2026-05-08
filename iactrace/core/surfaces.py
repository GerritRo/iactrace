from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .intersections import intersect_conic, newton_raphson_intersect

# Surface group protocol


class SurfaceGroup(eqx.Module):
    """Abstract base for batched surface parameters of N optical elements.

    A SurfaceGroup stores per-element surface geometry and provides:
    - Sag/normal computation for the transform pipeline (vmapped per element)
    - Per-element sag and ray intersection for rendering and visualization

    Subclasses must store an ``offsets`` array of shape (N, 2) and implement
    the abstract methods below.  See ``AsphericSurfaceGroup`` for a concrete
    example.
    """

    offsets: jax.Array  # (N, 2) — per-element surface offsets

    @abstractmethod
    def compute_sag_and_normal_at(self, x, y):
        """Compute surface point and normal at (x, y) for a single element.

        Designed to be called inside ``jax.vmap`` over the module's axis 0.
        After vmapping, per-element arrays become scalar/1D.

        Args:
            x: x-coordinate (scalar)
            y: y-coordinate (scalar)

        Returns:
            Tuple of (point, normal) where point is (3,) and normal is (3,).
        """
        ...

    @abstractmethod
    def sag_at(self, element_idx, x, y):
        """Compute surface sag z(x, y) for a single element.

        Used by the visualization module for mesh generation.

        Args:
            element_idx: Element index within the group.
            x: x-coordinate in local frame (scalar)
            y: y-coordinate in local frame (scalar)

        Returns:
            z: Surface sag at (x, y) relative to the element's offset.
        """
        ...

    @abstractmethod
    def intersect_at(self, element_idx, ray_origin, ray_direction,
                     max_iter=10, tol=1e-8):
        """Intersect a ray with a single element's surface.

        Used by the render pipeline for per-ray intersection.

        Args:
            element_idx: Element index within the group.
            ray_origin: Ray origin in local coordinates (3,)
            ray_direction: Ray direction (3,)
            max_iter: Maximum Newton-Raphson iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple of (t, point, normal):
                - t: Intersection distance (scalar)
                - point: Intersection point (3,)
                - normal: Surface normal at intersection (3,)
        """
        ...


def sag_raw(x, y, curvature, conic, aspheric):
    """Compute surface sag z(x,y) without offset.

    Args:
        x: x-coordinate (scalar)
        y: y-coordinate (scalar)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        z: Surface sag at (x, y)
    """
    r2 = x * x + y * y
    c = curvature
    k = conic

    denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
    z = r2 * c / denom

    if aspheric.size > 0:
        powers = jnp.arange(2, 2 + 2 * len(aspheric), 2)
        z = z + jnp.sum(aspheric * r2 ** powers)

    return z


def sag(x, y, offset, curvature, conic, aspheric):
    """Compute surface sag z(x,y) in local mirror coordinates.

    Args:
        x: x-coordinate in local mirror frame (scalar)
        y: y-coordinate in local mirror frame (scalar)
        offset: (x0, y0) offset on parent surface (2,)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        z: Surface sag at (x, y) relative to offset point
    """
    x0, y0 = offset[0], offset[1]
    z0 = sag_raw(x0, y0, curvature, conic, aspheric)
    return sag_raw(x + x0, y + y0, curvature, conic, aspheric) - z0


def compute_sag_and_normal(x, y, offset, curvature, conic, aspheric):
    """Compute surface point and normal at (x, y) with given parameters.

    Args:
        x: x-coordinate in local mirror frame (scalar)
        y: y-coordinate in local mirror frame (scalar)
        offset: (x0, y0) offset on parent surface (2,)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        point: 3D surface point (3,)
        normal: Surface normal (3,), normalized
    """
    z = sag(x, y, offset, curvature, conic, aspheric)
    point = jnp.stack([x, y, z], axis=-1)

    x_surf = x + offset[0]
    y_surf = y + offset[1]
    dzdx = jax.grad(lambda X: sag_raw(X, y_surf, curvature, conic, aspheric))(x_surf)
    dzdy = jax.grad(lambda Y: sag_raw(x_surf, Y, curvature, conic, aspheric))(y_surf)
    n = jnp.array([-dzdx, -dzdy, 1.0])
    normal = n / jnp.linalg.norm(n)

    return point, normal


#  Aspheric surface group


class AsphericSurfaceGroup(SurfaceGroup):
    """Batched aspheric surface parameters for N optical elements.

    This is the composable surface module used by ``OpticalElementGroup``.
    When vmapped over axis 0, each element becomes a single-element surface
    with scalar curvature/conic and (K,) aspheric coefficients, enabling
    generic use in transform_to_world and render pipelines.

    Attributes:
        curvatures: Per-element curvatures (N,)
        conics: Per-element conic constants (N,)
        aspherics: Per-element aspheric coefficients (N, K)
        offsets: Per-element surface offsets (N, 2)
    """

    curvatures: jax.Array   # (N,)
    conics: jax.Array        # (N,)
    aspherics: jax.Array     # (N, K)

    def compute_sag_and_normal_at(self, x, y):
        return compute_sag_and_normal(
            x, y, self.offsets, self.curvatures, self.conics, self.aspherics
        )

    def sag_at(self, element_idx, x, y):
        return sag(
            x, y, self.offsets[element_idx],
            self.curvatures[element_idx], self.conics[element_idx],
            self.aspherics[element_idx],
        )

    def intersect_at(self, element_idx, ray_origin, ray_direction,
                     max_iter=10, tol=1e-8):
        c = self.curvatures[element_idx]
        k = self.conics[element_idx]
        a = self.aspherics[element_idx]
        offset = self.offsets[element_idx]

        # Translate to raw (unshifted) surface frame for the conic initial guess
        z0 = sag_raw(offset[0], offset[1], c, k, a)
        ray_origin_raw = ray_origin + jnp.array([offset[0], offset[1], z0])

        t_init = intersect_conic(ray_origin_raw, ray_direction, c, k)

        # Refine with Newton-Raphson in the offset frame
        t, hit_xy, _ = newton_raphson_intersect(
            lambda x, y: sag(x, y, offset, c, k, a),
            ray_origin, ray_direction, t_init, max_iter, tol,
        )

        point, normal = compute_sag_and_normal(
            hit_xy[0], hit_xy[1], offset, c, k, a,
        )
        return t, point, normal

