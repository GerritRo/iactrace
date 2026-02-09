from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from .intersections import (
    intersect_box,
    intersect_cylinder,
    intersect_open_cylinder,
    intersect_oriented_box,
    intersect_sphere,
    intersect_triangle,
)
from .transforms import euler_to_matrix


class ObstructionGroup(eqx.Module):
    """Base class for grouped obstructions."""

    config_type: ClassVar[str] = ""  # Set by subclasses

    @abstractmethod
    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all primitives in group."""
        ...

    @abstractmethod
    def to_config(self, index: int) -> dict[str, Any]:
        """Convert a single obstruction at index to a config dict."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> ObstructionGroup:
        """Create an ObstructionGroup from a list of config dicts."""
        ...

    @abstractmethod
    def __len__(self):
        ...


class CylinderGroup(ObstructionGroup):
    config_type: ClassVar[str] = "cylinder"
    """Group of cylinders for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)
    r: jax.Array   # (N,)

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self):
        return self.p1.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all cylinders."""
        ts = vmap(intersect_cylinder, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2, self.r
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert cylinder at index to config dict."""
        return {
            "type": "cylinder",
            "p1": [float(x) for x in np.asarray(self.p1[index])],
            "p2": [float(x) for x in np.asarray(self.p2[index])],
            "r": float(self.r[index]),
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> CylinderGroup:
        """Create CylinderGroup from config dicts."""
        p1 = [c["p1"] for c in configs]
        p2 = [c["p2"] for c in configs]
        r = [c["r"] for c in configs]
        return cls(p1, p2, r)


class OpenCylinderGroup(ObstructionGroup):
    """Group of open cylinders (no end caps) for efficient batched intersection.

    An open cylinder is a finite cylindrical surface without circular caps at
    the ends. Useful for modeling tubes, pipes, or hollow cylindrical structures
    where rays can pass through the ends.
    """

    config_type: ClassVar[str] = "open_cylinder"

    p1: jax.Array  # (N, 3) - first endpoint of axis
    p2: jax.Array  # (N, 3) - second endpoint of axis
    r: jax.Array   # (N,) - radius

    def __init__(self, p1, p2, r):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.r = jnp.asarray(r)

    def __len__(self):
        return self.p1.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all open cylinders (curved surface only)."""
        ts = vmap(intersect_open_cylinder, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2, self.r
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert open cylinder at index to config dict."""
        return {
            "type": "open_cylinder",
            "p1": [float(x) for x in np.asarray(self.p1[index])],
            "p2": [float(x) for x in np.asarray(self.p2[index])],
            "r": float(self.r[index]),
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> OpenCylinderGroup:
        """Create OpenCylinderGroup from config dicts."""
        p1 = [c["p1"] for c in configs]
        p2 = [c["p2"] for c in configs]
        r = [c["r"] for c in configs]
        return cls(p1, p2, r)


class BoxGroup(ObstructionGroup):
    config_type: ClassVar[str] = "box"
    """Group of axis-aligned boxes for efficient batched intersection."""

    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)

    def __init__(self, p1, p2):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)

    def __len__(self):
        return self.p1.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all boxes."""
        ts = vmap(intersect_box, in_axes=(None, None, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert box at index to config dict."""
        return {
            "type": "box",
            "p1": [float(x) for x in np.asarray(self.p1[index])],
            "p2": [float(x) for x in np.asarray(self.p2[index])],
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> BoxGroup:
        """Create BoxGroup from config dicts."""
        p1 = [c["p1"] for c in configs]
        p2 = [c["p2"] for c in configs]
        return cls(p1, p2)


class SphereGroup(ObstructionGroup):
    config_type: ClassVar[str] = "sphere"
    """Group of spheres for efficient batched intersection."""

    centers: jax.Array  # (N, 3)
    radii: jax.Array    # (N,)

    def __init__(self, centers, radii):
        self.centers = jnp.asarray(centers)
        self.radii = jnp.asarray(radii)

    def __len__(self):
        return self.centers.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all spheres."""
        ts = vmap(intersect_sphere, in_axes=(None, None, 0, 0))(
            ray_origin, ray_direction, self.centers, self.radii
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert sphere at index to config dict."""
        return {
            "type": "sphere",
            "center": [float(x) for x in np.asarray(self.centers[index])],
            "r": float(self.radii[index]),
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> SphereGroup:
        """Create SphereGroup from config dicts."""
        centers = [c["center"] for c in configs]
        radii = [c["r"] for c in configs]
        return cls(centers, radii)


class OrientedBoxGroup(ObstructionGroup):
    config_type: ClassVar[str] = "oriented_box"
    """Group of oriented boxes for efficient batched intersection."""

    centers: jax.Array       # (N, 3)
    half_extents: jax.Array  # (N, 3)
    rotations: jax.Array     # (N, 3, 3)

    def __init__(self, centers, half_extents, rotations):
        self.centers = jnp.asarray(centers)
        self.half_extents = jnp.asarray(half_extents)
        self.rotations = jnp.asarray(rotations)

    def __len__(self):
        return self.centers.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all oriented boxes."""
        ts = vmap(intersect_oriented_box, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.centers, self.half_extents, self.rotations
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert oriented box at index to config dict."""
        rotation_matrix = np.asarray(self.rotations[index])
        euler = _rotation_matrix_to_euler(rotation_matrix)
        return {
            "type": "oriented_box",
            "center": [float(x) for x in np.asarray(self.centers[index])],
            "half_extents": [float(x) for x in np.asarray(self.half_extents[index])],
            "rotation": euler,
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> OrientedBoxGroup:
        """Create OrientedBoxGroup from config dicts."""
        centers = [c["center"] for c in configs]
        half_extents = [c["half_extents"] for c in configs]
        # Convert euler angles to rotation matrices
        rotations = []
        for c in configs:
            euler = jnp.asarray(c["rotation"])
            rot_matrix = euler_to_matrix(euler)
            rotations.append(rot_matrix)
        return cls(centers, half_extents, jnp.stack(rotations))


class TriangleGroup(ObstructionGroup):
    config_type: ClassVar[str] = "triangle"
    """Group of triangles for efficient batched intersection."""

    v0: jax.Array  # (N, 3)
    v1: jax.Array  # (N, 3)
    v2: jax.Array  # (N, 3)

    def __init__(self, v0, v1, v2):
        self.v0 = jnp.asarray(v0)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)

    def __len__(self):
        return self.v0.shape[0]

    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all triangles."""
        ts = vmap(intersect_triangle, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.v0, self.v1, self.v2
        )
        return jnp.min(ts)

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert triangle at index to config dict."""
        return {
            "type": "triangle",
            "v0": [float(x) for x in np.asarray(self.v0[index])],
            "v1": [float(x) for x in np.asarray(self.v1[index])],
            "v2": [float(x) for x in np.asarray(self.v2[index])],
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> TriangleGroup:
        """Create TriangleGroup from config dicts."""
        v0 = [c["v0"] for c in configs]
        v1 = [c["v1"] for c in configs]
        v2 = [c["v2"] for c in configs]
        return cls(v0, v1, v2)


def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to Euler angles (degrees).

    Uses the same convention as euler_to_matrix from transforms.py.

    Args:
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        List of [rx, ry, rz] Euler angles in degrees.
    """
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

    if sy > 1e-6:
        rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        rx = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = 0.0

    return [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]
