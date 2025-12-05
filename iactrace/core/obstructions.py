import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from jax import vmap

from .intersections import (
    intersect_cylinder,
    intersect_box,
    intersect_sphere,
    intersect_oriented_box,
    intersect_triangle,
)


class Obstruction(eqx.Module):
    """Base class for single obstructions."""
    
    @abstractmethod
    def intersect(self, ray_origin, ray_direction):
        pass


class ObstructionGroup(eqx.Module):
    """Base class for grouped obstructions."""
    
    @abstractmethod
    def intersect(self, ray_origin, ray_direction):
        """Returns min t across all primitives in group."""
        pass
    
    def __len__(self):
        raise NotImplementedError


# === Cylinder ===

class Cylinder(Obstruction):
    """Single cylindrical obstruction."""
    
    p1: jax.Array
    p2: jax.Array
    radius: float = eqx.field(static=True)
    
    def __init__(self, p1, p2, radius):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
        self.radius = float(radius)
    
    def intersect(self, ray_origin, ray_direction):
        return intersect_cylinder(ray_origin, ray_direction, self.p1, self.p2, self.radius)


class CylinderGroup(ObstructionGroup):
    """Group of cylinders for efficient batched intersection."""
    
    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)
    r: jax.Array   # (N,)
    
    def __init__(self, cylinders=None, p1=None, p2=None, r=None):
        if cylinders is not None:
            self.p1 = jnp.stack([c.p1 for c in cylinders])
            self.p2 = jnp.stack([c.p2 for c in cylinders])
            self.r = jnp.array([c.radius for c in cylinders])
        else:
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


# === Box (axis-aligned) ===

class Box(Obstruction):
    """Single axis-aligned box obstruction."""
    
    p1: jax.Array
    p2: jax.Array
    
    def __init__(self, p1, p2):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
    
    def intersect(self, ray_origin, ray_direction):
        return intersect_box(ray_origin, ray_direction, self.p1, self.p2)


class BoxGroup(ObstructionGroup):
    """Group of axis-aligned boxes for efficient batched intersection."""
    
    p1: jax.Array  # (N, 3)
    p2: jax.Array  # (N, 3)
    
    def __init__(self, boxes=None, p1=None, p2=None):
        if boxes is not None:
            self.p1 = jnp.stack([b.p1 for b in boxes])
            self.p2 = jnp.stack([b.p2 for b in boxes])
        else:
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


# === Sphere ===

class Sphere(Obstruction):
    """Single spherical obstruction."""
    
    center: jax.Array
    radius: float = eqx.field(static=True)
    
    def __init__(self, center, radius):
        self.center = jnp.asarray(center)
        self.radius = float(radius)
    
    def intersect(self, ray_origin, ray_direction):
        return intersect_sphere(ray_origin, ray_direction, self.center, self.radius)


class SphereGroup(ObstructionGroup):
    """Group of spheres for efficient batched intersection."""
    
    centers: jax.Array  # (N, 3)
    radii: jax.Array    # (N,)
    
    def __init__(self, spheres=None, centers=None, radii=None):
        if spheres is not None:
            self.centers = jnp.stack([s.center for s in spheres])
            self.radii = jnp.array([s.radius for s in spheres])
        else:
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


# === Oriented Box ===

class OrientedBox(Obstruction):
    """Single oriented box obstruction."""
    
    center: jax.Array
    half_extents: jax.Array
    rotation: jax.Array  # (3, 3) rotation matrix
    
    def __init__(self, center, half_extents, rotation):
        self.center = jnp.asarray(center)
        self.half_extents = jnp.asarray(half_extents)
        self.rotation = jnp.asarray(rotation)
    
    def intersect(self, ray_origin, ray_direction):
        return intersect_oriented_box(
            ray_origin, ray_direction, self.center, self.half_extents, self.rotation
        )


class OrientedBoxGroup(ObstructionGroup):
    """Group of oriented boxes for efficient batched intersection."""
    
    centers: jax.Array       # (N, 3)
    half_extents: jax.Array  # (N, 3)
    rotations: jax.Array     # (N, 3, 3)
    
    def __init__(self, boxes=None, centers=None, half_extents=None, rotations=None):
        if boxes is not None:
            self.centers = jnp.stack([b.center for b in boxes])
            self.half_extents = jnp.stack([b.half_extents for b in boxes])
            self.rotations = jnp.stack([b.rotation for b in boxes])
        else:
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


# === Triangle ===

class Triangle(Obstruction):
    """Single triangle obstruction."""
    
    v0: jax.Array
    v1: jax.Array
    v2: jax.Array
    
    def __init__(self, v0, v1, v2):
        self.v0 = jnp.asarray(v0)
        self.v1 = jnp.asarray(v1)
        self.v2 = jnp.asarray(v2)
    
    def intersect(self, ray_origin, ray_direction):
        return intersect_triangle(ray_origin, ray_direction, self.v0, self.v1, self.v2)


class TriangleGroup(ObstructionGroup):
    """Group of triangles for efficient batched intersection."""
    
    v0: jax.Array  # (N, 3)
    v1: jax.Array  # (N, 3)
    v2: jax.Array  # (N, 3)
    
    def __init__(self, triangles=None, v0=None, v1=None, v2=None):
        if triangles is not None:
            self.v0 = jnp.stack([t.v0 for t in triangles])
            self.v1 = jnp.stack([t.v1 for t in triangles])
            self.v2 = jnp.stack([t.v2 for t in triangles])
        else:
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


def group_obstructions(obstructions):
    """Convert list of obstructions to list of groups by type."""
    cylinders = [o for o in obstructions if isinstance(o, Cylinder)]
    boxes = [o for o in obstructions if isinstance(o, Box)]
    spheres = [o for o in obstructions if isinstance(o, Sphere)]
    oriented_boxes = [o for o in obstructions if isinstance(o, OrientedBox)]
    triangles = [o for o in obstructions if isinstance(o, Triangle)]
    
    groups = []
    if cylinders:
        groups.append(CylinderGroup(cylinders))
    if boxes:
        groups.append(BoxGroup(boxes))
    if spheres:
        groups.append(SphereGroup(spheres))
    if oriented_boxes:
        groups.append(OrientedBoxGroup(oriented_boxes))
    if triangles:
        groups.append(TriangleGroup(triangles))
    
    return groups