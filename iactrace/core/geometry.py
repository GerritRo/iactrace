import jax
import jax.numpy as jnp
from jax import vmap

### Intersection tests

def intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation):
    """
    Intersect ray with a plane defined by center and rotation matrix.
    
    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        plane_center: Plane center (3,)
        plane_rotation: Rotation matrix (3, 3) - Z-axis is normal
    
    Returns:
        2D coordinates on plane (2,)
    """
    # Extract normal and basis vectors from rotation matrix
    u1 = plane_rotation[:, 0]  # X-axis in plane coordinates
    u2 = plane_rotation[:, 1]  # Y-axis in plane coordinates
    plane_normal = plane_rotation[:, 2]  # Z-axis = normal
    
    # Find t parameter for intersection
    ndotd = jnp.sum(ray_direction * plane_normal, axis=-1)
    ndoto = jnp.sum(ray_origin * plane_normal, axis=-1)
    ndotp = jnp.sum(plane_normal * plane_center)
    t = (ndotp - ndoto) / ndotd
    
    # Intersection point
    intersection = ray_origin + t[..., None] * ray_direction
    
    # Project onto plane coordinate system
    op = intersection - plane_center
    x = jnp.sum(op * u1, axis=-1)
    y = jnp.sum(op * u2, axis=-1)
    
    return jnp.stack([x, y], axis=-1)


def intersect_cylinder(ray_origin, ray_direction, p1, p2, radius):
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


def intersect_box(ray_origin, ray_direction, p1, p2):
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


### Normals manipulation

def perturb_normals(normals, sigma_rad, key):
    """
    Perturb normals by random angles.
    
    Args:
        normals: Surface normals (..., 3), assumed unit length
        sigma_rad: RMS perturbation angle in radians
        key: JAX random key
    
    Returns:
        Perturbed unit normals (..., 3)
    """
    shape = normals.shape[:-1]
    
    key1, key2 = jax.random.split(key)
    theta1 = jax.random.normal(key1, shape) * sigma_rad
    theta2 = jax.random.normal(key2, shape) * sigma_rad
    
    # Build tangent basis, avoiding degeneracy
    ref_z = jnp.array([0., 0., 1.])
    ref_x = jnp.array([1., 0., 0.])
    
    dot_z = jnp.abs(jnp.sum(normals * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)
    
    tangent1 = jnp.cross(normals, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(normals, tangent1)
    
    perturbed = normals + theta1[..., None] * tangent1 + theta2[..., None] * tangent2
    return perturbed / jnp.linalg.norm(perturbed, axis=-1, keepdims=True)