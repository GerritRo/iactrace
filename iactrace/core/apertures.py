from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def _point_in_convex_polygon(x, y, vertices, n_vertices):
    """Check if points (x, y) are inside a convex polygon.

    For CCW vertices, a point is inside if all edge cross products are >= 0.
    For CW  vertices, a point is inside if all edge cross products are <= 0.

    Args:
        x: x-coordinates (can be scalar or array)
        y: y-coordinates (can be scalar or array)
        vertices: Polygon vertices in CCW order (K, 2)
        n_vertices: Number of vertices

    Returns:
        Boolean mask, True if inside polygon
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
    """Compute area of convex polygon using shoelace formula.

    Args:
        vertices: Polygon vertices (K, 2)

    Returns:
        Polygon area (scalar)
    """
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

    Attributes:
        radii: Outer radius per element (N,)
        inner_radii: Inner radius per element (N,), 0 for solid disk
    """

    radii: Array  # (N,)
    inner_radii: Array  # (N,)

    def check(self, x, y, element_idx):
        """Check if point (x, y) is within the aperture of the given element."""
        r_sq = x**2 + y**2
        return (r_sq >= self.inner_radii[element_idx] ** 2) & (r_sq <= self.radii[element_idx] ** 2)

    def sample(self, key, n_samples):
        """Sample uniform 2D points on each element's annular aperture.

        Args:
            key: JAX PRNG key
            n_samples: Number of samples per element

        Returns:
            (N, n_samples, 2) array of 2D sample points
        """
        from .sampling import sample_annulus

        keys = jax.random.split(key, self.radii.shape[0])
        return jax.vmap(
            lambda k, inner_r, outer_r: sample_annulus(k, inner_r, outer_r, (n_samples,))
        )(keys, self.inner_radii, self.radii)

    def get_area_data(self):
        """Return per-element data for area computation, vmapped over elements.

        Returns:
            Array of shape (N, 2) with [inner_radius, outer_radius] per element.
        """
        return jnp.stack([self.inner_radii, self.radii], axis=-1)

    def area_fn(self, data):
        """Compute aperture area from a single element's area data.

        Args:
            data: [inner_radius, outer_radius] (2,)

        Returns:
            Annular area (scalar)
        """
        return jnp.pi * (data[1] ** 2 - data[0] ** 2)


class PolygonAperture(Aperture):
    """Convex polygon aperture defined by vertices.

    All elements in a group must have the same number of vertices
    (required for JAX array batching).

    Attributes:
        vertices: Polygon vertices per element (N, K, 2), CCW order
        n_vertices: Number of vertices per polygon (static, same for all)
    """

    vertices: Array  # (N, K, 2)
    n_vertices: int = eqx.field(static=True)

    def check(self, x, y, element_idx):
        """Check if point (x, y) is within the polygon aperture."""
        return _point_in_convex_polygon(x, y, self.vertices[element_idx], self.n_vertices)

    def sample(self, key, n_samples):
        """Sample uniform 2D points on each element's polygon aperture.

        Args:
            key: JAX PRNG key
            n_samples: Number of samples per element

        Returns:
            (N, n_samples, 2) array of 2D sample points
        """
        from .sampling import sample_polygon

        keys = jax.random.split(key, self.vertices.shape[0])
        return jax.vmap(lambda k, verts: sample_polygon(k, verts, (n_samples,)))(
            keys, self.vertices
        )

    def get_area_data(self):
        """Return per-element data for area computation, vmapped over elements.

        Returns:
            Vertices array (N, K, 2).
        """
        return self.vertices

    def area_fn(self, data):
        """Compute polygon area from a single element's vertices.

        Args:
            data: Polygon vertices (K, 2)

        Returns:
            Polygon area (scalar)
        """
        return _polygon_area(data)
