from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .intersections import (
    intersect_box,
    intersect_cylinder,
    intersect_open_cylinder,
    intersect_oriented_box,
    intersect_sphere,
    intersect_triangle,
)

_VMAP_PAIR_BUDGET = 2_800_000


class ObstructionGroup(eqx.Module):
    """Base class for grouped obstructions.

    Subclasses supply their intersection kernel and stacked parameters via
    :meth:`_primitive`; the two traversal strategies are shared from here.
    """

    @abstractmethod
    def _primitive(self):
        """Return ``(kernel, params)`` for this group.

        ``kernel(ray_origin, ray_direction, *prim)`` intersects one ray with one
        primitive and returns a scalar ``t`` (``inf`` on a miss). ``params`` is a
        tuple of stacked parameter arrays whose leading axis indexes the
        primitives.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of obstruction primitives in this group."""
        ...

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all primitives in group."""
        kernel, params = self._primitive()
        ts = jax.vmap(kernel, in_axes=(None, None) + (0,) * len(params))(
            ray_origin, ray_direction, *params
        )
        return jnp.min(ts)

    def intersect_batch(self, origins, directions):
        """Nearest hit distance per ray, for ``(n_rays, 3)`` rays.

        Same answer as ``vmap(self.intersect)``, but chooses how to walk the
        primitives based on how many rays there are.
        """
        if origins.shape[0] * len(self) <= _VMAP_PAIR_BUDGET:
            return jax.vmap(self.intersect)(origins, directions)

        kernel, params = self._primitive()
        dtype = jnp.result_type(origins, directions, *params)
        batched = jax.vmap(kernel, in_axes=(0, 0) + (None,) * len(params))

        def step(best_t, prim):
            t = batched(origins, directions, *prim)
            return jnp.minimum(best_t, t.astype(dtype)), None

        best_t, _ = jax.lax.scan(
            step, jnp.full(origins.shape[0], jnp.inf, dtype=dtype), params
        )
        return best_t


class CylinderGroup(ObstructionGroup):
    """Group of cylinders for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)
    r: jax.Array  # (N,)

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self) -> int:
        return self.r.shape[0]

    def _primitive(self):
        return intersect_cylinder, (self.p1, self.p2, self.r)


class OpenCylinderGroup(ObstructionGroup):
    """Group of open cylinders (no end caps) for efficient batched intersection.

    An open cylinder is a finite cylindrical surface without circular caps at
    the ends. Useful for modeling tubes, pipes, or hollow cylindrical structures
    where rays can pass through the ends.
    """

    p1: jax.Array  # (N, 3) - first endpoint of axis
    p2: jax.Array  # (N, 3) - second endpoint of axis
    r: jax.Array  # (N,) - radius

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self) -> int:
        return self.r.shape[0]

    def _primitive(self):
        return intersect_open_cylinder, (self.p1, self.p2, self.r)


class BoxGroup(ObstructionGroup):
    """Group of axis-aligned boxes for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)

    def __init__(self, p1, p2):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)

    def __len__(self) -> int:
        return self.p1.shape[0]

    def _primitive(self):
        return intersect_box, (self.p1, self.p2)


class SphereGroup(ObstructionGroup):
    """Group of spheres for efficient batched intersection."""

    centers: jax.Array  # (N, 3)
    radii: jax.Array  # (N,)

    def __init__(self, centers, radii):
        self.centers = jnp.asarray(centers)
        self.radii = jnp.asarray(radii)

    def __len__(self) -> int:
        return self.radii.shape[0]

    def _primitive(self):
        return intersect_sphere, (self.centers, self.radii)


class OrientedBoxGroup(ObstructionGroup):
    """Group of oriented boxes for efficient batched intersection."""

    centers: jax.Array  # (N, 3)
    half_extents: jax.Array  # (N, 3)
    rotations: jax.Array  # (N, 3, 3)

    def __init__(self, centers, half_extents, rotations):
        self.centers = jnp.asarray(centers)
        self.half_extents = jnp.asarray(half_extents)
        self.rotations = jnp.asarray(rotations)

    def __len__(self) -> int:
        return self.centers.shape[0]

    def _primitive(self):
        return intersect_oriented_box, (self.centers, self.half_extents, self.rotations)


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

    def _primitive(self):
        return intersect_triangle, (self.v0, self.v1, self.v2)
