from __future__ import annotations

import jax
import jax.numpy as jnp

from .tolerances import dir_tol, len_rel

# Placeholder in-plane coordinates for a ray that never meets the plane.
_MISS_PLANE_XY = 1e10

# How far past the derived tolerance a Newton residual may sit and still count as converged.
_ACCEPT_FACTOR = 8.0


def is_hit(t):
    """Whether a ray parameter denotes a real intersection."""
    return jnp.isfinite(t)


# Numerical helpers


def _safe_divide(numerator, denominator, fallback=jnp.inf):
    """numerator / denominator, yielding fallback where the denominator is 0."""
    ok = denominator != 0.0
    return jnp.where(ok, numerator / jnp.where(ok, denominator, 1.0), fallback)


def _safe_sqrt(x):
    """sqrt(x) for positive x and zero elsewhere, with a finite gradient."""
    ok = x > 0.0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)


def _finite(t):
    """t with a non-finite value replaced by zero, so inf * 0 cannot arise."""
    return jnp.where(jnp.isfinite(t), t, 0.0)


def _localize(ray_origin, ray_direction, reference, scale=0.0):
    """Move ray origin along the ray to its closest approach to reference.

    Parameters
    ----------
    ray_origin : array, shape (3,)
        Ray origin.
    ray_direction : array, shape (3,)
        Ray direction, assumed normalized.
    reference : array, shape (3,)
        Point to move towards, usually the primitive's centre.
    scale
        Size of the primitive.

    Returns
    -------
    (origin, t_offset, t_lo): the shifted origin, the ray parameter that
    reaches it, and the smallest local t distinguishable from a re-hit of
    the surface the ray just left.
    """
    ox, oy, oz = ray_origin[0], ray_origin[1], ray_origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]

    t_offset = (reference[0] - ox) * dx + (reference[1] - oy) * dy + (reference[2] - oz) * dz
    px = ox + t_offset * dx
    py = oy + t_offset * dy
    pz = oz + t_offset * dz
    origin = jnp.stack([px, py, pz], axis=-1)

    norm = jnp.sqrt(px * px + py * py + pz * pz)
    t_floor = len_rel(origin) * (jnp.abs(t_offset) + norm + scale)
    return origin, t_offset, t_floor - t_offset


def _to_world(t_offset, t_local):
    """Undo the shift of _localize, keeping a miss exactly inf."""
    hit = jnp.isfinite(t_local)
    return jnp.where(hit, t_offset + jnp.where(hit, t_local, 0.0), jnp.inf)


def _quadratic_roots(a, half_b, c):
    """Both roots of a t^2 + 2 half_b t + c, without the cancelling subtraction.

    Parameters
    ----------
    a, half_b, c
        Coefficients, with the linear one already halved.

    Returns
    -------
    (t1, t2, real). The roots come non-ordered.
    """
    disc = half_b * half_b - a * c
    sign = jnp.where(half_b < 0.0, -1.0, 1.0)
    q = -(half_b + sign * _safe_sqrt(disc))
    return _safe_divide(q, a), _safe_divide(c, q), disc >= 0.0


def _slab_hit(origin, direction, lo, hi, t_lo):
    """Nearest acceptable hit of an axis-aligned box, by the slab method.

    Parameters
    ----------
    origin, direction
        Ray expressed in the box's own frame.
    lo, hi : array, shape (3,)
        Opposite corners, with lo <= hi componentwise.
    t_lo
        Smallest acceptable ray parameter.

    Returns
    -------
    The hit parameter, inf if the ray misses.
    """
    crosses = jnp.abs(direction) > dir_tol(direction)
    safe_dir = jnp.where(crosses, direction, 1.0)
    t1 = (lo - origin) / safe_dir
    t2 = (hi - origin) / safe_dir

    # A slab the ray runs parallel to either holds it for every t or for none.
    inside = (origin >= lo) & (origin <= hi)
    t_near = jnp.where(crosses, jnp.minimum(t1, t2), jnp.where(inside, -jnp.inf, jnp.inf))
    t_far = jnp.where(crosses, jnp.maximum(t1, t2), jnp.where(inside, jnp.inf, -jnp.inf))

    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)
    # Entry face from outside, exit face for a ray that starts within the box.
    t_hit = jnp.where(t_min > t_lo, t_min, t_max)
    return jnp.where((t_max >= t_min) & (t_max > t_lo), t_hit, jnp.inf)


def _cylinder_frame(origin, ray_direction, p1, axis):
    """Split the ray into components along and across a cylinder axis."""
    oc = origin - p1
    oc_axial = jnp.dot(oc, axis)
    rd_axial = jnp.dot(ray_direction, axis)
    return oc_axial, rd_axial, oc - oc_axial * axis, ray_direction - rd_axial * axis


def _cylinder_side(oc_axial, rd_axial, oc_perp, rd_perp, height, radius, t_lo):
    """The two hits of the curved surface of a finite cylinder, inf where absent."""
    a = jnp.dot(rd_perp, rd_perp)
    half_b = jnp.dot(oc_perp, rd_perp)
    c = jnp.dot(oc_perp, oc_perp) - radius * radius
    t1, t2, real = _quadratic_roots(a, half_b, c)

    crosses = real & (a > dir_tol(a) ** 2)
    y1 = oc_axial + _finite(t1) * rd_axial
    y2 = oc_axial + _finite(t2) * rd_axial
    ok1 = crosses & (t1 > t_lo) & (y1 >= 0.0) & (y1 <= height)
    ok2 = crosses & (t2 > t_lo) & (y2 >= 0.0) & (y2 <= height)
    return jnp.where(ok1, t1, jnp.inf), jnp.where(ok2, t2, jnp.inf)


# --- Primitive intersections ---


def intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation):
    """Intersect a ray with a plane, and locate the hit within it.

    Unlike the other kernels this returns (xy, t): the in-plane coordinates
    of the hit along plane_rotation's first two columns (its third is the
    normal), and the ray parameter. A ray that never meets the plane gets
    t = inf and xy = _MISS_PLANE_XY.
    """
    u1 = plane_rotation[:, 0]
    u2 = plane_rotation[:, 1]
    plane_normal = plane_rotation[:, 2]
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, plane_center)

    ndotd = jnp.dot(ray_direction, plane_normal)
    parallel = jnp.abs(ndotd) < dir_tol(ndotd)
    safe_ndotd = jnp.where(parallel, 1.0, ndotd)
    # Differencing the two points before projecting them avoids subtracting a
    # pair of nearly equal projections when the plane sits far from the origin.
    t_local = jnp.dot(plane_center - origin, plane_normal) / safe_ndotd

    op = origin + t_local * ray_direction - plane_center
    valid = ~parallel & (t_local > t_lo)
    xy = jnp.where(valid, jnp.array([jnp.dot(op, u1), jnp.dot(op, u2)]), _MISS_PLANE_XY)
    return xy, _to_world(t_offset, jnp.where(valid, t_local, jnp.inf))


def intersect_cylinder(ray_origin, ray_direction, p1, p2, radius):
    """Intersect ray with a finite cylinder, end caps included.

    Parameters
    ----------
    p1 : array, shape (3,)
        First endpoint of cylinder axis.
    p2 : array, shape (3,)
        Second endpoint of cylinder axis.
    radius : scalar
        Cylinder radius.
    """
    axis = p2 - p1
    height = jnp.linalg.norm(axis)
    axis = axis / height
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, 0.5 * (p1 + p2), height)

    oc_axial, rd_axial, oc_perp, rd_perp = _cylinder_frame(origin, ray_direction, p1, axis)
    t1, t2 = _cylinder_side(oc_axial, rd_axial, oc_perp, rd_perp, height, radius, t_lo)

    # The caps are the planes at axial 0 and height, clipped to the radius.
    caps_ok = jnp.abs(rd_axial) > dir_tol(rd_axial)
    safe_axial = jnp.where(caps_ok, rd_axial, 1.0)
    t_bottom = -oc_axial / safe_axial
    t_top = (height - oc_axial) / safe_axial
    perp_bottom = oc_perp + t_bottom * rd_perp
    perp_top = oc_perp + t_top * rd_perp
    r2 = radius * radius
    t_bottom = jnp.where(
        caps_ok & (t_bottom > t_lo) & (jnp.dot(perp_bottom, perp_bottom) <= r2), t_bottom, jnp.inf
    )
    t_top = jnp.where(
        caps_ok & (t_top > t_lo) & (jnp.dot(perp_top, perp_top) <= r2), t_top, jnp.inf
    )

    return _to_world(t_offset, jnp.minimum(jnp.minimum(t1, t2), jnp.minimum(t_bottom, t_top)))


def intersect_open_cylinder(ray_origin, ray_direction, p1, p2, radius):
    """Intersect ray with finite cylinder without end caps (curved surface only).

    Parameters
    ----------
    p1 : array, shape (3,)
        First endpoint of cylinder axis.
    p2 : array, shape (3,)
        Second endpoint of cylinder axis.
    radius : scalar
        Cylinder radius.
    """
    axis = p2 - p1
    height = jnp.linalg.norm(axis)
    axis = axis / height
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, 0.5 * (p1 + p2), height)

    oc_axial, rd_axial, oc_perp, rd_perp = _cylinder_frame(origin, ray_direction, p1, axis)
    t1, t2 = _cylinder_side(oc_axial, rd_axial, oc_perp, rd_perp, height, radius, t_lo)
    return _to_world(t_offset, jnp.minimum(t1, t2))


def intersect_box(ray_origin, ray_direction, p1, p2):
    """Intersect ray with AABB box.

    Parameters
    ----------
    p1 : array, shape (3,)
        lower edge of the bounding box.
    p2 : array, shape (3,)
        upper diagonal edge of the bounding box.
    """
    box_min = jnp.minimum(p1, p2)
    box_max = jnp.maximum(p1, p2)
    half = 0.5 * jnp.max(box_max - box_min)
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, 0.5 * (box_min + box_max), half)
    return _to_world(t_offset, _slab_hit(origin, ray_direction, box_min, box_max, t_lo))


def intersect_oriented_box(ray_origin, ray_direction, center, half_extents, rotation):
    """Intersect ray with oriented bounding box.

    Parameters
    ----------
    center : array, shape (3,)
        Box center.
    half_extents : array, shape (3,)
        Half-sizes along local axes.
    rotation : array, shape (3, 3)
        Rotation matrix transforming local to world coordinates.
    """
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, center, jnp.max(half_extents))
    rot_inv = rotation.T
    local_origin = rot_inv @ (origin - center)
    local_direction = rot_inv @ ray_direction
    t_local = _slab_hit(local_origin, local_direction, -half_extents, half_extents, t_lo)
    return _to_world(t_offset, t_local)


def intersect_triangle(ray_origin, ray_direction, v0, v1, v2):
    """Intersect ray with triangle using Moeller-Trumbore algorithm.

    Parameters
    ----------
    v0, v1, v2 : array, shape (3,)
        Triangle vertices.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    size = jnp.maximum(jnp.linalg.norm(edge1), jnp.linalg.norm(edge2))
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, (v0 + v1 + v2) / 3.0, size)

    h = jnp.cross(ray_direction, edge2)
    a = jnp.dot(edge1, h)
    parallel = jnp.abs(a) <= dir_tol(a) * jnp.linalg.norm(edge1) * jnp.linalg.norm(h)
    f = jnp.where(parallel, 0.0, 1.0 / jnp.where(parallel, 1.0, a))

    s = origin - v0
    q = jnp.cross(s, edge1)
    u = f * jnp.dot(s, h)
    v = f * jnp.dot(ray_direction, q)
    t = f * jnp.dot(edge2, q)

    inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    return _to_world(t_offset, jnp.where(~parallel & inside & (t > t_lo), t, jnp.inf))


def intersect_sphere(ray_origin, ray_direction, center, radius):
    """Intersect ray with sphere.

    Parameters
    ----------
    center : array, shape (3,)
        Sphere center.
    radius : scalar
        Sphere radius.
    """
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, center, radius)
    oc = origin - center
    gap = radius * radius - jnp.dot(oc, oc)
    t_half = _safe_sqrt(gap)
    hits = gap >= 0.0
    t1 = jnp.where(hits & (-t_half > t_lo), -t_half, jnp.inf)
    t2 = jnp.where(hits & (t_half > t_lo), t_half, jnp.inf)
    return _to_world(t_offset, jnp.minimum(t1, t2))


def intersect_conic(ray_origin, ray_direction, curvature, conic):
    """Closed-form ray-conic intersection, on the sag branch only.

    Parameters
    ----------
    curvature
        Surface curvature (1/radius)
    conic
        Conic constant (0=sphere, -1=paraboloid, <-1=hyperboloid,
        >-1=ellipsoid)
    """
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, jnp.zeros(3))
    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]
    c = curvature
    k = conic

    # From substituting the ray into c*(x^2 + y^2) + (1 + k)*c*z^2 - 2*z = 0.
    a = c * (dx * dx + dy * dy + (1 + k) * dz * dz)
    half_b = c * (ox * dx + oy * dy + (1 + k) * oz * dz) - dz
    const = c * (ox * ox + oy * oy + (1 + k) * oz * oz) - 2 * oz
    t1, t2, real = _quadratic_roots(a, half_b, const)

    # Keep the sag branch only.
    z1 = oz + _finite(t1) * dz
    z2 = oz + _finite(t2) * dz
    branch_slack = 1.0 + dir_tol(z1)
    ok1 = real & (t1 > t_lo) & ((1 + k) * c * z1 <= branch_slack)
    ok2 = real & (t2 > t_lo) & ((1 + k) * c * z2 <= branch_slack)

    t_conic = jnp.minimum(jnp.where(ok1, t1, jnp.inf), jnp.where(ok2, t2, jnp.inf))
    return _to_world(t_offset, t_conic)


# Newton-Raphson method


def newton_raphson_intersect(sag_fn, ray_origin, ray_direction, t_init=None, max_iter=10, tol=None):
    """Ray-surface intersection by Newton-Raphson, for any sag function.

    The generic fallback for a surface with no closed-form root. Rays are in
    the surface's local coordinates.

    Parameters
    ----------
    sag_fn
        Callable (x, y) -> z giving surface height.
    t_init
        Initial guess; None starts from the z = 0 plane.
    max_iter
        Maximum iterations.
    tol
        Absolute residual tolerance on z - sag(x, y), in the
        coordinates' own units. None derives it per ray.

    Returns
    -------
    (t, hit_xy, valid): the ray parameter (inf if no intersection),
    the (2,) in-surface coordinates of the hit, and whether it
    converged to a real intersection.
    """
    origin, t_offset, t_lo = _localize(ray_origin, ray_direction, jnp.zeros(3))
    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]
    local_scale = jnp.abs(t_offset) + jnp.linalg.norm(origin)

    # Initial guess: use provided value or intersect with z=0 plane.
    if t_init is None:
        dz_ok = jnp.abs(dz) > dir_tol(dz)
        t_init = jnp.where(dz_ok, -oz / jnp.where(dz_ok, dz, 1.0), 0.0)
    else:
        t_init = t_init - t_offset

    def g(t):
        """Implicit function: g(t) = 0 at intersection."""
        return oz + t * dz - sag_fn(ox + t * dx, oy + t * dy)

    def newton_step(carry, _):
        t, converged = carry
        g_val, g_prime = jax.value_and_grad(g)(t)
        slope_ok = jnp.abs(g_prime) > dir_tol(g_prime)
        step = jnp.where(slope_ok, g_val / jnp.where(slope_ok, g_prime, 1.0), 0.0)
        settled = jnp.abs(step) <= len_rel(t) * (jnp.abs(t) + local_scale)
        return (jnp.where(converged, t, t - step), converged | settled), None

    (t_final, _), _ = jax.lax.scan(newton_step, (t_init, False), None, length=max_iter)

    x_hit = ox + t_final * dx
    y_hit = oy + t_final * dy
    z_hit = oz + t_final * dz
    sag = sag_fn(x_hit, y_hit)
    residual = jnp.abs(z_hit - sag)
    if tol is None:
        scale = jnp.abs(x_hit) + jnp.abs(y_hit) + jnp.abs(z_hit) + jnp.abs(sag)
        tol = len_rel(z_hit) * scale

    valid = (t_final > t_lo) & (residual <= _ACCEPT_FACTOR * tol)
    t_out = _to_world(t_offset, jnp.where(valid, t_final, jnp.inf))
    return t_out, jnp.array([x_hit, y_hit]), valid
