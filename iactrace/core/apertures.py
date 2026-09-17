from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .sampling import sample_annulus, sample_polygon


def _point_in_convex_polygon(x, y, vertices, n_vertices):
    """Boolean mask: are the points (x, y) inside the convex polygon?

    Accepts either winding -- a point is inside if every edge cross product is
    >= 0 (CCW) or every one is <= 0 (CW).
    """

    def edge_check(carry, i):
        ccw, cw = carry
        v1, v2 = vertices[i], vertices[(i + 1) % n_vertices]
        cross = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0])
        return (ccw & (cross >= 0), cw & (cross <= 0)), None

    ones = jnp.ones_like(x, dtype=bool)
    (ccw, cw), _ = jax.lax.scan(edge_check, (ones, ones), jnp.arange(n_vertices))
    return ccw | cw


def _polygon_area(vertices):
    """Area of the convex polygon with vertices (K, 2), by the shoelace formula."""
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    return 0.5 * jnp.abs(jnp.sum(vx * jnp.roll(vy, -1) - jnp.roll(vx, -1) * vy))


class Aperture(eqx.Module):
    """Abstract base for aperture modules."""

    @abstractmethod
    def check(self, x, y, element_idx): ...

    @abstractmethod
    def sample(self, key, n_samples): ...

    @abstractmethod
    def get_area_data(self): ...

    @abstractmethod
    def area_fn(self, data): ...


class DiskAperture(Aperture):
    """Circular or annular aperture defined by outer and inner radii.

    Supports solid disks (inner_radii=0) and annular rings.

    Attributes
    ----------
    radii : array, shape (N,)
        Outer radius per element.
    inner_radii : array, shape (N,)
        Inner radius per element, 0 for a solid disk.
    """

    radii: Array  # (N,)
    inner_radii: Array  # (N,)

    def check(self, x, y, element_idx):
        r_sq = x**2 + y**2
        return (r_sq >= self.inner_radii[element_idx] ** 2) & (r_sq <= self.radii[element_idx] ** 2)

    def sample(self, key, n_samples):
        """(N, n_samples, 2) uniform points on each element's annulus."""
        keys = jax.random.split(key, self.radii.shape[0])
        return jax.vmap(
            lambda k, inner_r, outer_r: sample_annulus(k, inner_r, outer_r, (n_samples,))
        )(keys, self.inner_radii, self.radii)

    def get_area_data(self):
        """(N, 2) of [inner_radius, outer_radius] per element."""
        return jnp.stack([self.inner_radii, self.radii], axis=-1)

    def area_fn(self, data):
        """Annular area from one element's [inner_radius, outer_radius]."""
        return jnp.pi * (data[1] ** 2 - data[0] ** 2)


class PolygonAperture(Aperture):
    """Convex polygon aperture defined by vertices.

    All elements in a group must have the same number of vertices.

    Attributes
    ----------
    vertices : array, shape (N, K, 2)
        Polygon vertices per element, CCW order.
    n_vertices : array, shape (static, same for all)
        Number of vertices per polygon.
    """

    vertices: Array  # (N, K, 2)
    n_vertices: int = eqx.field(static=True)

    def check(self, x, y, element_idx):
        return _point_in_convex_polygon(x, y, self.vertices[element_idx], self.n_vertices)

    def sample(self, key, n_samples):
        """(N, n_samples, 2) uniform points on each element's polygon."""
        keys = jax.random.split(key, self.vertices.shape[0])
        return jax.vmap(lambda k, verts: sample_polygon(k, verts, (n_samples,)))(
            keys, self.vertices
        )

    def get_area_data(self):
        """The (N, K, 2) vertices; area needs nothing else."""
        return self.vertices

    def area_fn(self, data):
        """Polygon area from one element's (K, 2) vertices."""
        return _polygon_area(data)
