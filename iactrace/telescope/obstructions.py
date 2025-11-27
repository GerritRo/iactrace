import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from jax import vmap


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

def _cylinder_intersect(ray_origin, ray_direction, p1, p2, radius):
    """Single cylinder intersection (for vmapping)."""
    axis = p2 - p1
    height = jnp.linalg.norm(axis)
    axis = axis / height
    
    oc = ray_origin - p1
    oc_axial = jnp.dot(oc, axis)
    rd_axial = jnp.dot(ray_direction, axis)
    oc_perp = oc - oc_axial * axis
    rd_perp = ray_direction - rd_axial * axis
    
    a = jnp.dot(rd_perp, rd_perp)
    b = 2 * jnp.dot(oc_perp, rd_perp)
    c = jnp.dot(oc_perp, oc_perp) - radius * radius
    disc = b * b - 4 * a * c
    
    eps = 1e-8
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a + eps)
    t2 = (-b + sqrt_disc) / (2 * a + eps)
    
    y1 = oc_axial + t1 * rd_axial
    y2 = oc_axial + t2 * rd_axial
    
    t1 = jnp.where((t1 > eps) & (y1 >= 0) & (y1 <= height) & (disc >= 0), t1, jnp.inf)
    t2 = jnp.where((t2 > eps) & (y2 >= 0) & (y2 <= height) & (disc >= 0), t2, jnp.inf)
    
    t_bottom = -oc_axial / (rd_axial + eps)
    t_top = (height - oc_axial) / (rd_axial + eps)
    
    perp_bottom = oc_perp + t_bottom * rd_perp
    perp_top = oc_perp + t_top * rd_perp
    
    t_bottom = jnp.where(
        (t_bottom > eps) & (jnp.dot(perp_bottom, perp_bottom) <= radius**2),
        t_bottom, jnp.inf
    )
    t_top = jnp.where(
        (t_top > eps) & (jnp.dot(perp_top, perp_top) <= radius**2),
        t_top, jnp.inf
    )
    
    return jnp.min(jnp.array([t1, t2, t_bottom, t_top]))


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
        return _cylinder_intersect(ray_origin, ray_direction, self.p1, self.p2, self.radius)


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
        ts = vmap(_cylinder_intersect, in_axes=(None, None, 0, 0, 0))(
            ray_origin, ray_direction, self.p1, self.p2, self.r
        )
        return jnp.min(ts)


# === Box ===

def _box_intersect(ray_origin, ray_direction, p1, p2):
    """Single box intersection (for vmapping)."""
    eps = 1e-8
    
    box_min = jnp.minimum(p1, p2)
    box_max = jnp.maximum(p1, p2)
    
    inv_dir = 1.0 / (ray_direction + eps)
    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir
    
    t_near = jnp.minimum(t1, t2)
    t_far = jnp.maximum(t1, t2)
    
    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)
    
    hit = (t_max >= t_min) & (t_max > eps)
    t_result = jnp.where(t_min > eps, t_min, t_max)
    
    return jnp.where(hit, t_result, jnp.inf)


class Box(Obstruction):
    """Single axis-aligned box obstruction."""
    
    p1: jax.Array
    p2: jax.Array
    
    def __init__(self, p1, p2):
        self.p1 = jnp.asarray(p1)
        self.p2 = jnp.asarray(p2)
    
    def intersect(self, ray_origin, ray_direction):
        return _box_intersect(ray_origin, ray_direction, self.p1, self.p2)


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
        ts = vmap(_box_intersect, in_axes=(None, None, 0, 0))(
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
