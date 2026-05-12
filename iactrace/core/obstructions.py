from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap

from .intersections import (
    intersect_box,
    intersect_cylinder,
    intersect_open_cylinder,
    intersect_oriented_box,
    intersect_sphere,
    intersect_triangle,
)


class ObstructionGroup(eqx.Module):
    """Base class for grouped obstructions."""

    @abstractmethod
    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all primitives in group."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of obstruction primitives in this group."""
        ...


class CylinderGroup(ObstructionGroup):
    """Group of cylinders for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)
    r: jax.Array   # (N,)

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self) -> int:
        return self.r.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all cylinders."""
        ts = vmap(intersect_cylinder, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2, self.r
        )
        return jnp.min(ts)


class OpenCylinderGroup(ObstructionGroup):
    """Group of open cylinders (no end caps) for efficient batched intersection.

    An open cylinder is a finite cylindrical surface without circular caps at
    the ends. Useful for modeling tubes, pipes, or hollow cylindrical structures
    where rays can pass through the ends.
    """

    p1: jax.Array  # (N, 3) - first endpoint of axis
    p2: jax.Array  # (N, 3) - second endpoint of axis
    r: jax.Array   # (N,) - radius

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self) -> int:
        return self.r.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all open cylinders (curved surface only)."""
        ts = vmap(intersect_open_cylinder, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2, self.r
        )
        return jnp.min(ts)


class BoxGroup(ObstructionGroup):
    """Group of axis-aligned boxes for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)

    def __init__(self, p1, p2):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)

    def __len__(self) -> int:
        return self.p1.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all boxes."""
        ts = vmap(intersect_box, in_axes=(None, None, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2
        )
        return jnp.min(ts)


class SphereGroup(ObstructionGroup):
    """Group of spheres for efficient batched intersection."""

    centers: jax.Array  # (N, 3)
    radii: jax.Array    # (N,)

    def __init__(self, centers, radii):
        self.centers = jnp.asarray(centers)
        self.radii = jnp.asarray(radii)

    def __len__(self) -> int:
        return self.radii.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all spheres."""
        ts = vmap(intersect_sphere, in_axes=(None, None, 0, 0))(
            ray_origin, ray_direction, self.centers, self.radii
        )
        return jnp.min(ts)


class OrientedBoxGroup(ObstructionGroup):
    """Group of oriented boxes for efficient batched intersection."""

    centers: jax.Array       # (N, 3)
    half_extents: jax.Array  # (N, 3)
    rotations: jax.Array     # (N, 3, 3)

    def __init__(self, centers, half_extents, rotations):
        self.centers = jnp.asarray(centers)
        self.half_extents = jnp.asarray(half_extents)
        self.rotations = jnp.asarray(rotations)

    def __len__(self) -> int:
        return self.centers.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all oriented boxes."""
        ts = vmap(intersect_oriented_box, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.centers, self.half_extents, self.rotations
        )
        return jnp.min(ts)


class TriangleGroup(ObstructionGroup):
    """Group of triangles for efficient batched intersection."""

    v0: jax.Array  # (N, 3)
    v1: jax.Array  # (N, 3)
    v2: jax.Array  # (N, 3)

    def __init__(self, v0, v1, v2):
        self.v0 = jnp.asarray(v0)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)

    def __len__(self) -> int:
        return self.v0.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all triangles."""
        ts = vmap(intersect_triangle, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.v0, self.v1, self.v2
        )
        return jnp.min(ts)
