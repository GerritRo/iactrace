import jax.numpy as jnp
from jax import vmap


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

def intersect_cylinder(ray_origin, ray_direction, point1, point2, radius):
    """
    Intersect a ray with a finite cylinder (with caps).

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        point1: Cylinder start point (3,)
        point2: Cylinder end point (3,)
        radius: Cylinder radius (scalar)

    Returns:
        Distance to nearest intersection (scalar), inf if no hit
    """
    axis = point2 - point1
    height = jnp.linalg.norm(axis)
    axis = axis / height

    oc = ray_origin - point1

    # Project onto axis and perpendicular plane
    oc_axial = jnp.dot(oc, axis)
    rd_axial = jnp.dot(ray_direction, axis)
    oc_perp = oc - oc_axial * axis
    rd_perp = ray_direction - rd_axial * axis

    # Quadratic for infinite cylinder
    a = jnp.dot(rd_perp, rd_perp)
    b = 2 * jnp.dot(oc_perp, rd_perp)
    c = jnp.dot(oc_perp, oc_perp) - radius * radius
    disc = b * b - 4 * a * c

    eps = 1e-8

    # Cylindrical surface
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a + eps)
    t2 = (-b + sqrt_disc) / (2 * a + eps)

    y1 = oc_axial + t1 * rd_axial
    y2 = oc_axial + t2 * rd_axial

    # Check if between point1 and point2
    t1 = jnp.where((t1 > eps) & (y1 >= 0) & (y1 <= height) & (disc >= 0), t1, jnp.inf)
    t2 = jnp.where((t2 > eps) & (y2 >= 0) & (y2 <= height) & (disc >= 0), t2, jnp.inf)

    # Caps at point1 and point2
    t_bottom = (0 - oc_axial) / (rd_axial + eps)
    t_top = (height - oc_axial) / (rd_axial + eps)

    perp_bottom = oc_perp + t_bottom * rd_perp
    perp_top = oc_perp + t_top * rd_perp

    t_bottom = jnp.where((t_bottom > eps) & (jnp.dot(perp_bottom, perp_bottom) <= radius*radius),
                         t_bottom, jnp.inf)
    t_top = jnp.where((t_top > eps) & (jnp.dot(perp_top, perp_top) <= radius*radius),
                      t_top, jnp.inf)

    return jnp.min(jnp.array([t1, t2, t_bottom, t_top]))


def intersect_box(ray_origin, ray_direction, point1, point2):
    """
    Intersect a ray with an axis-aligned box.

    The box is defined by two diagonal corner points (p1 and p2).

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        point1: First corner of box (3,)
        point2: Opposite diagonal corner (3,)

    Returns:
        Distance to nearest intersection (scalar), inf if no hit
    """
    eps = 1e-8

    # Box bounds (handle case where p1 > p2 for some coordinates)
    box_min = jnp.minimum(point1, point2)
    box_max = jnp.maximum(point1, point2)

    # Compute intersection distances for each slab
    inv_dir = 1.0 / (ray_direction + eps)
    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir

    # Get near and far intersection for each axis
    t_near = jnp.minimum(t1, t2)
    t_far = jnp.maximum(t1, t2)

    # Overall near and far
    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)

    # Check if there's a valid intersection
    hit = (t_max >= t_min) & (t_max > eps)
    t_result = jnp.where(t_min > eps, t_min, t_max)

    return jnp.where(hit, t_result, jnp.inf)


def check_occlusions(ray_origins, ray_directions, 
                           cyl_p1=None, cyl_p2=None, cyl_radius=None,
                           box_p1=None, box_p2=None):
    """
    Check if rays are occluded by cylinders and boxes.

    Returns:
        Shadow mask (N, M) - 1.0 if not occluded, 0.0 if occluded
    """
    shadow_mask = jnp.ones(ray_origins.shape[:-1])
    
    if len(cyl_p1) > 0:
        def min_cylinder_t(origin, direction):
            """Find minimum t across all cylinders."""
            ts = vmap(intersect_cylinder, in_axes=(None, None, 0, 0, 0))(
                origin, direction, cyl_p1, cyl_p2, cyl_radius)
            return jnp.min(ts)

        # Check cylinder occlusions
        t_cylinders = vmap(vmap(min_cylinder_t, in_axes=(0, 0)), in_axes=(0, 0))(
            ray_origins, ray_directions)

        shadow_mask = shadow_mask * jnp.where(t_cylinders < 1e10, 0.0, 1.0)

    if len(box_p1) > 0:
        def min_box_t(origin, direction):
            """Find minimum t across all boxes."""
            ts = vmap(intersect_box, in_axes=(None, None, 0, 0))(
                origin, direction, box_p1, box_p2)
            return jnp.min(ts)

        # Check box occlusions
        t_boxes = vmap(vmap(min_box_t, in_axes=(0, 0)), in_axes=(0, 0))(
            ray_origins, ray_directions)

        shadow_mask = shadow_mask * jnp.where(t_boxes < 1e10, 0.0, 1.0)

    return shadow_mask
