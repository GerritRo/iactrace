import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from jax import vmap

from .geometry import intersect_cylinder, intersect_box

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


# === Box ===

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
    """Group of boxes for efficient batched intersection."""
    
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


def group_obstructions(obstructions):
    """Convert list of obstructions to list of groups by type."""
    cyls = [o for o in obstructions if isinstance(o, Cylinder)]
    boxes = [o for o in obstructions if isinstance(o, Box)]
    
    groups = []
    if cyls:
        groups.append(CylinderGroup(cyls))
    if boxes:
        groups.append(BoxGroup(boxes))
    
    return groups
