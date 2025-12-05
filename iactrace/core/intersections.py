import jax
import jax.numpy as jnp
from jax import vmap

### Primitive intersections

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
    u1 = plane_rotation[:, 0]
    u2 = plane_rotation[:, 1]
    plane_normal = plane_rotation[:, 2]
    
    ndotd = jnp.sum(ray_direction * plane_normal, axis=-1)
    ndoto = jnp.sum(ray_origin * plane_normal, axis=-1)
    ndotp = jnp.sum(plane_normal * plane_center)
    
    parallel = jnp.abs(ndotd) < 1e-10
    safe_ndotd = jnp.where(parallel, 1.0, ndotd)
    t = (ndotp - ndoto) / safe_ndotd
    
    intersection = ray_origin + t[..., None] * ray_direction
    
    op = intersection - plane_center
    x = jnp.sum(op * u1, axis=-1)
    y = jnp.sum(op * u2, axis=-1)
    
    invalid = parallel | (t <= 0)
    x = jnp.where(invalid, 1e10, x)
    y = jnp.where(invalid, 1e10, y)
    
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


def intersect_oriented_box(ray_origin, ray_direction, center, half_extents, rotation):
    """
    Intersect ray with oriented bounding box.
    
    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        center: Box center (3,)
        half_extents: Half-sizes along local axes (3,)
        rotation: Rotation matrix (3, 3) transforming local to world coords
    
    Returns:
        t parameter of nearest intersection, jnp.inf if no hit
    """
    eps = 1e-8
    
    # Transform ray to box's local coordinate system
    rot_inv = rotation.T
    local_origin = rot_inv @ (ray_origin - center)
    local_direction = rot_inv @ ray_direction
    
    # Standard AABB test in local coords
    inv_dir = 1.0 / (local_direction + eps * jnp.sign(local_direction + eps))
    
    t1 = (-half_extents - local_origin) * inv_dir
    t2 = (half_extents - local_origin) * inv_dir
    
    t_near = jnp.minimum(t1, t2)
    t_far = jnp.maximum(t1, t2)
    
    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)
    
    hit = (t_max >= t_min) & (t_max > eps)
    t_result = jnp.where(t_min > eps, t_min, t_max)
    
    return jnp.where(hit & (t_result > eps), t_result, jnp.inf)


def intersect_triangle(ray_origin, ray_direction, v0, v1, v2):
    """
    Intersect ray with triangle using Möller-Trumbore algorithm.
    
    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        v0, v1, v2: Triangle vertices (3,) each
    
    Returns:
        t parameter of intersection, jnp.inf if no hit
    """
    eps = 1e-8
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = jnp.cross(ray_direction, edge2)
    a = jnp.dot(edge1, h)
    
    # Ray parallel to triangle
    parallel = jnp.abs(a) < eps
    
    f = 1.0 / (a + eps * jnp.sign(a + eps))
    s = ray_origin - v0
    u = f * jnp.dot(s, h)
    
    q = jnp.cross(s, edge1)
    v = f * jnp.dot(ray_direction, q)
    
    t = f * jnp.dot(edge2, q)
    
    # Check validity: not parallel, barycentric coords valid, t > 0
    valid = (
        ~parallel &
        (u >= 0.0) & (u <= 1.0) &
        (v >= 0.0) & (u + v <= 1.0) &
        (t > eps)
    )
    
    return jnp.where(valid, t, jnp.inf)


def intersect_sphere(ray_origin, ray_direction, center, radius):
    """
    Intersect ray with sphere.
    
    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,), assumed normalized
        center: Sphere center (3,)
        radius: Sphere radius (scalar)
    
    Returns:
        t parameter of nearest intersection, jnp.inf if no hit
    """
    eps = 1e-8
    oc = ray_origin - center
    
    # Quadratic coefficients: |O + t*D - C|^2 = r^2
    a = jnp.dot(ray_direction, ray_direction)
    b = 2.0 * jnp.dot(oc, ray_direction)
    c = jnp.dot(oc, oc) - radius * radius
    
    disc = b * b - 4.0 * a * c
    
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a + eps)
    t2 = (-b + sqrt_disc) / (2.0 * a + eps)
    
    # Return nearest positive intersection
    t1_valid = jnp.where((t1 > eps) & (disc >= 0), t1, jnp.inf)
    t2_valid = jnp.where((t2 > eps) & (disc >= 0), t2, jnp.inf)
    
    return jnp.minimum(t1_valid, t2_valid)


def intersect_conic(ray_origin, ray_direction, curvature, conic):
    """
    Compute closed-form ray-conic intersection parameter.

    The conic surface is defined by the implicit equation:
        c*(x² + y²) + (1+k)*c*z² - 2*z = 0

    where c is curvature and k is the conic constant.

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,), assumed normalized
        curvature: Surface curvature (1/radius)
        conic: Conic constant (0=sphere, -1=paraboloid, <-1=hyperboloid, >-1=ellipsoid)

    Returns:
        t: Ray parameter at intersection (smallest positive root), inf if no intersection
    """
    ox, oy, oz = ray_origin[0], ray_origin[1], ray_origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]
    c = curvature
    k = conic

    # Quadratic coefficients: A*t² + B*t + C = 0
    # From substituting ray into: c*(x² + y²) + (1+k)*c*z² - 2*z = 0
    A = c * (dx * dx + dy * dy + (1 + k) * dz * dz)
    B = 2 * (c * (ox * dx + oy * dy + (1 + k) * oz * dz) - dz)
    C = c * (ox * ox + oy * oy + (1 + k) * oz * oz) - 2 * oz

    # Handle near-zero curvature (plane)
    is_plane = jnp.abs(c) < 1e-12
    t_plane = jnp.where(jnp.abs(dz) > 1e-10, -oz / dz, jnp.inf)

    # Solve quadratic
    discriminant = B * B - 4 * A * C

    # No real roots
    no_intersection = discriminant < 0

    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))

    # Two roots
    t1 = (-B - sqrt_disc) / (2 * A + 1e-30)
    t2 = (-B + sqrt_disc) / (2 * A + 1e-30)

    # Select smallest positive root
    t1_valid = t1 > 1e-8
    t2_valid = t2 > 1e-8

    t_conic = jnp.where(
        t1_valid & t2_valid,
        jnp.minimum(t1, t2),
        jnp.where(t1_valid, t1, jnp.where(t2_valid, t2, jnp.inf))
    )
    t_conic = jnp.where(no_intersection, jnp.inf, t_conic)

    return jnp.where(is_plane, t_plane, t_conic)


### Newton-Raphson method

def newton_raphson_intersect(sag_fn, ray_origin, ray_direction, t_init=None, max_iter=10, tol=1e-8):
    """
    Find ray-surface intersection using Newton-Raphson iteration.

    This is a generic intersection routine for any surface defined by a sag function z = f(x, y).

    Args:
        sag_fn: Callable (x, y) -> z giving surface height
        ray_origin: Ray origin in local coordinates (3,)
        ray_direction: Ray direction in local coordinates (3,), assumed normalized
        t_init: Initial guess for ray parameter. If None, uses z=0 plane intersection.
        max_iter: Maximum Newton-Raphson iterations
        tol: Convergence tolerance

    Returns:
        t: Parameter along ray (scalar), inf if no intersection
        hit_xy: (x, y) coordinates at intersection (2,)
        valid: Boolean indicating if intersection is valid
    """
    ox, oy, oz = ray_origin[0], ray_origin[1], ray_origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]

    # Initial guess: use provided value or intersect with z=0 plane
    if t_init is None:
        t_init = jnp.where(
            jnp.abs(dz) > 1e-10,
            -oz / dz,
            0.0
        )
        t_init = jnp.maximum(t_init, 0.0)

    def g(t):
        """Implicit function: g(t) = 0 at intersection."""
        x = ox + t * dx
        y = oy + t * dy
        z = oz + t * dz
        return z - sag_fn(x, y)

    def g_deriv(t):
        """Derivative dg/dt using autodiff."""
        return jax.grad(g)(t)

    def newton_step(carry, _):
        t, converged = carry
        g_val = g(t)
        g_prime = g_deriv(t)

        # Avoid division by zero
        g_prime_safe = jnp.where(jnp.abs(g_prime) > 1e-12, g_prime, 1e-12)
        t_new = t - g_val / g_prime_safe

        # Check convergence
        new_converged = converged | (jnp.abs(g_val) < tol)

        # Only update if not converged
        t_out = jnp.where(converged, t, t_new)

        return (t_out, new_converged), None

    (t_final, _), _ = jax.lax.scan(
        newton_step,
        (t_init, False),
        None,
        length=max_iter
    )

    # Compute hit coordinates
    x_hit = ox + t_final * dx
    y_hit = oy + t_final * dy
    hit_xy = jnp.array([x_hit, y_hit])

    # Check validity: t should be positive and residual small
    residual = jnp.abs(g(t_final))
    valid = (t_final > 1e-8) & (residual < tol * 100)

    t_out = jnp.where(valid, t_final, jnp.inf)

    return t_out, hit_xy, valid
