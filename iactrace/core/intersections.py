import jax
import jax.numpy as jnp

### Primitive intersections


def _reorigin(ray_origin, ray_direction, reference):
    """
    Slide the ray origin along the ray to its closest approach to ``reference``.
    """
    t_offset = jnp.dot(reference - ray_origin, ray_direction)
    return ray_origin + t_offset * ray_direction, t_offset


def intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation):
    """
    Intersect ray with a plane defined by center and rotation matrix.

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        plane_center: Plane center (3,)
        plane_rotation: Rotation matrix (3, 3) - Z-axis is normal

    Returns:
        Tuple of (2D coordinates on plane (2,), t parameter (scalar))
    """
    u1 = plane_rotation[:, 0]
    u2 = plane_rotation[:, 1]
    plane_normal = plane_rotation[:, 2]

    # Solve on a ray origin slid next to the plane (see _reorigin). ``t`` itself
    # is well conditioned, but ``origin + t * direction`` is not: for a distant
    # source it cancels two vectors of order the source distance to land on a
    # focal plane whose interesting structure is millimetres across.
    t_offset = jnp.sum((plane_center - ray_origin) * ray_direction, axis=-1)
    origin = ray_origin + t_offset[..., None] * ray_direction

    ndotd = jnp.sum(ray_direction * plane_normal, axis=-1)
    ndoto = jnp.sum(origin * plane_normal, axis=-1)
    ndotp = jnp.sum(plane_normal * plane_center)

    parallel = jnp.abs(ndotd) < 1e-10
    safe_ndotd = jnp.where(parallel, 1.0, ndotd)
    t_local = (ndotp - ndoto) / safe_ndotd
    t = t_offset + t_local

    intersection = origin + t_local[..., None] * ray_direction

    op = intersection - plane_center
    x = jnp.sum(op * u1, axis=-1)
    y = jnp.sum(op * u2, axis=-1)

    invalid = parallel | (t <= 0)
    x = jnp.where(invalid, 1e10, x)
    y = jnp.where(invalid, 1e10, y)
    t = jnp.where(invalid, jnp.inf, t)

    return jnp.stack([x, y], axis=-1), t


def intersect_cylinder(ray_origin, ray_direction, p1, p2, radius):
    """Single cylinder intersection (for vmapping)."""
    eps = 1e-8
    axis = p2 - p1
    height = jnp.linalg.norm(axis)
    axis = axis / height

    # Work next to the cylinder, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, 0.5 * (p1 + p2))
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"

    oc = origin - p1
    oc_axial = jnp.dot(oc, axis)
    rd_axial = jnp.dot(ray_direction, axis)
    oc_perp = oc - oc_axial * axis
    rd_perp = ray_direction - rd_axial * axis

    a = jnp.dot(rd_perp, rd_perp)
    b = 2 * jnp.dot(oc_perp, rd_perp)
    c = jnp.dot(oc_perp, oc_perp) - radius * radius
    disc = b * b - 4 * a * c

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a + eps)
    t2 = (-b + sqrt_disc) / (2 * a + eps)

    y1 = oc_axial + t1 * rd_axial
    y2 = oc_axial + t2 * rd_axial

    # A ray parallel to the axis never crosses the curved surface: there a and b
    # both vanish and the roots collapse to 0/eps, which the forward test used to
    # discard only incidentally.
    crosses = (a > eps) & (disc >= 0)
    t1 = jnp.where((t1 > t_lo) & (y1 >= 0) & (y1 <= height) & crosses, t1, jnp.inf)
    t2 = jnp.where((t2 > t_lo) & (y2 >= 0) & (y2 <= height) & crosses, t2, jnp.inf)

    t_bottom = -oc_axial / (rd_axial + eps)
    t_top = (height - oc_axial) / (rd_axial + eps)

    perp_bottom = oc_perp + t_bottom * rd_perp
    perp_top = oc_perp + t_top * rd_perp

    t_bottom = jnp.where(
        (t_bottom > t_lo) & (jnp.dot(perp_bottom, perp_bottom) <= radius**2), t_bottom, jnp.inf
    )
    t_top = jnp.where((t_top > t_lo) & (jnp.dot(perp_top, perp_top) <= radius**2), t_top, jnp.inf)

    return t_offset + jnp.min(jnp.array([t1, t2, t_bottom, t_top]))


def intersect_open_cylinder(ray_origin, ray_direction, p1, p2, radius):
    """
    Intersect ray with finite cylinder without end caps (curved surface only).

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,), assumed normalized
        p1: First endpoint of cylinder axis (3,)
        p2: Second endpoint of cylinder axis (3,)
        radius: Cylinder radius (scalar)

    Returns:
        t parameter of nearest intersection, jnp.inf if no hit
    """
    eps = 1e-8
    axis = p2 - p1
    height = jnp.linalg.norm(axis)
    axis = axis / height

    # Work next to the cylinder, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, 0.5 * (p1 + p2))
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"

    oc = origin - p1
    oc_axial = jnp.dot(oc, axis)
    rd_axial = jnp.dot(ray_direction, axis)
    oc_perp = oc - oc_axial * axis
    rd_perp = ray_direction - rd_axial * axis

    a = jnp.dot(rd_perp, rd_perp)
    b = 2 * jnp.dot(oc_perp, rd_perp)
    c = jnp.dot(oc_perp, oc_perp) - radius * radius
    disc = b * b - 4 * a * c

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a + eps)
    t2 = (-b + sqrt_disc) / (2 * a + eps)

    # Check if intersection points are within the finite cylinder height
    y1 = oc_axial + t1 * rd_axial
    y2 = oc_axial + t2 * rd_axial

    # A ray parallel to the axis never crosses the curved surface (see
    # intersect_cylinder): a and b both vanish and the roots collapse to 0/eps.
    crosses = (a > eps) & (disc >= 0)
    t1 = jnp.where((t1 > t_lo) & (y1 >= 0) & (y1 <= height) & crosses, t1, jnp.inf)
    t2 = jnp.where((t2 > t_lo) & (y2 >= 0) & (y2 <= height) & crosses, t2, jnp.inf)

    # Return nearest valid intersection (curved surface only, no caps)
    return t_offset + jnp.minimum(t1, t2)


def intersect_box(ray_origin, ray_direction, p1, p2):
    """
    Intersect ray with AABB box

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,)
        p1: lower edge of the bounding box (3,)
        p2: upper diagonal edge of the bounding box (3,)

    Returns:
        t parameter of nearest intersection, jnp.inf if no hit
    """
    eps = 1e-8

    box_min = jnp.minimum(p1, p2)
    box_max = jnp.maximum(p1, p2)

    # Work next to the box, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, 0.5 * (box_min + box_max))
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"

    inv_dir = 1.0 / (ray_direction + eps)
    t1 = (box_min - origin) * inv_dir
    t2 = (box_max - origin) * inv_dir

    t_near = jnp.minimum(t1, t2)
    t_far = jnp.maximum(t1, t2)

    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)

    hit = (t_max >= t_min) & (t_max > t_lo)
    t_result = jnp.where(t_min > t_lo, t_min, t_max)

    return jnp.where(hit, t_offset + t_result, jnp.inf)


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

    # Work next to the box, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, center)
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"

    # Transform ray to box's local coordinate system
    rot_inv = rotation.T
    local_origin = rot_inv @ (origin - center)
    local_direction = rot_inv @ ray_direction

    # Standard AABB test in local coords
    inv_dir = 1.0 / (local_direction + eps * jnp.sign(local_direction + eps))

    t1 = (-half_extents - local_origin) * inv_dir
    t2 = (half_extents - local_origin) * inv_dir

    t_near = jnp.minimum(t1, t2)
    t_far = jnp.maximum(t1, t2)

    t_min = jnp.max(t_near)
    t_max = jnp.min(t_far)

    hit = (t_max >= t_min) & (t_max > t_lo)
    t_result = jnp.where(t_min > t_lo, t_min, t_max)

    return jnp.where(hit & (t_result > t_lo), t_offset + t_result, jnp.inf)


def intersect_triangle(ray_origin, ray_direction, v0, v1, v2):
    """
    Intersect ray with triangle using Moeller-Trumbore algorithm.

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

    # Work next to the triangle, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, (v0 + v1 + v2) / 3.0)
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"

    h = jnp.cross(ray_direction, edge2)
    a = jnp.dot(edge1, h)

    # Ray parallel to triangle
    parallel = jnp.abs(a) < eps

    f = 1.0 / (a + eps * jnp.sign(a + eps))
    s = origin - v0
    u = f * jnp.dot(s, h)

    q = jnp.cross(s, edge1)
    v = f * jnp.dot(ray_direction, q)

    t = f * jnp.dot(edge2, q)

    # Check validity: not parallel, barycentric coords valid, t > 0
    valid = ~parallel & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_lo)

    return jnp.where(valid, t_offset + t, jnp.inf)


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

    # Work next to the sphere, not at the source's distance (see _reorigin).
    origin, t_offset = _reorigin(ray_origin, ray_direction, center)
    t_lo = eps - t_offset  # local bound equivalent to "total t > eps"
    oc = origin - center

    # Quadratic coefficients: |O + t*D - C|^2 = r^2
    a = jnp.dot(ray_direction, ray_direction)
    b = 2.0 * jnp.dot(oc, ray_direction)
    c = jnp.dot(oc, oc) - radius * radius

    disc = b * b - 4.0 * a * c

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2.0 * a + eps)
    t2 = (-b + sqrt_disc) / (2.0 * a + eps)

    # Return nearest positive intersection
    t1_valid = jnp.where((t1 > t_lo) & (disc >= 0), t1, jnp.inf)
    t2_valid = jnp.where((t2 > t_lo) & (disc >= 0), t2, jnp.inf)

    return t_offset + jnp.minimum(t1_valid, t2_valid)


def intersect_conic(ray_origin, ray_direction, curvature, conic):
    """
    Compute closed-form ray-conic intersection parameter.

    The conic surface is defined by the implicit equation:
        c*(x^2 + y^2) + (1+k)*c*z^2 - 2*z = 0

    where c is curvature and k is the conic constant.

    That implicit surface is closed for ``(1 + k) * c != 0`` -- a sphere
    (``k = 0``) is the full ball of radius ``R = 1/c`` centred on ``(0, 0, R)``,
    spanning ``z`` in ``[0, 2R]`` -- so a ray can cross it twice, on two
    different sheets. Only one of them is the optical surface: solving the conic
    for ``z`` at fixed ``r`` gives
    ``z = (1 -/+ sqrt(1 - (1 + k) c^2 r^2)) / ((1 + k) c)``, and the sag branch is
    the minus sign, i.e. the sheet through the vertex satisfying
    ``(1 + k) * c * z <= 1``. The other sheet is the far side of the ball, which
    no sag function describes.

    The nearest forward crossing is therefore *not* the answer in general: a ray
    starting beyond ``z = 2R / (1 + k)`` enters through the far sheet first. This
    returns the nearest forward crossing **that lies on the sag branch**.

    The ray is re-origined onto the vertex first (see :func:`_reorigin`): the
    quadratic's ``C`` grows as ``c * oz^2`` while ``B^2`` grows as ``4 c^2 oz^2``,
    so ``B^2 - 4AC`` cancels two terms of order ``(c * source_distance)^2`` down
    to something set by the aperture. Sliding the origin onto the vertex first
    keeps those at the scale of the optic. Absolute coordinates are unchanged by
    the slide, so the sag-branch test below still reads a true ``z``.

    Args:
        ray_origin: Ray origin (3,)
        ray_direction: Ray direction (3,), assumed normalized
        curvature: Surface curvature (1/radius)
        conic: Conic constant (0=sphere, -1=paraboloid, <-1=hyperboloid, >-1=ellipsoid)

    Returns:
        t: Ray parameter at the nearest forward intersection on the sag branch,
        inf if there is none.
    """
    origin, t_offset = _reorigin(ray_origin, ray_direction, jnp.zeros(3))
    t_lo = 1e-8 - t_offset  # local bound equivalent to "total t > 1e-8"

    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]
    c = curvature
    k = conic

    # Quadratic coefficients: A*t^2 + B*t + C = 0
    # From substituting ray into: c*(x^2 + y^2) + (1+k)*c*z^2 - 2*z = 0
    A = c * (dx * dx + dy * dy + (1 + k) * dz * dz)
    B = 2 * (c * (ox * dx + oy * dy + (1 + k) * oz * dz) - dz)
    C = c * (ox * ox + oy * oy + (1 + k) * oz * oz) - 2 * oz

    # Handle near-zero curvature (plane). Grad-safe division (double-where):
    # a bare -oz/dz would inject nan into the gradient for rays parallel to
    # the plane (dz == 0), even though the inf branch is the one selected.
    is_plane = jnp.abs(c) < 1e-12
    dz_ok = jnp.abs(dz) > 1e-10
    t_plane = jnp.where(dz_ok, -oz / jnp.where(dz_ok, dz, 1.0), jnp.inf)
    t_plane = jnp.where(t_plane > t_lo, t_plane, jnp.inf)

    # Handle linear case (A ~ 0, e.g., paraboloid on-axis); grad-safe as above.
    is_linear = jnp.abs(A) < 1e-12
    b_ok = jnp.abs(B) > 1e-12
    t_linear = jnp.where(b_ok, -C / jnp.where(b_ok, B, 1.0), jnp.inf)
    t_linear = jnp.where(t_linear > t_lo, t_linear, jnp.inf)

    # Solve quadratic. Grad-safe sqrt: at discriminant == 0 (e.g. the fully
    # degenerate flat-plane case A == B == 0) the sqrt's infinite slope would
    # otherwise nan the gradient even though the plane branch is selected.
    discriminant = B * B - 4 * A * C
    no_intersection = discriminant < 0
    disc_ok = discriminant > 0.0
    sqrt_disc = jnp.where(disc_ok, jnp.sqrt(jnp.where(disc_ok, discriminant, 1.0)), 0.0)

    # Two roots. Guard the division the same way (double-where) rather than by
    # nudging the denominator: with A == 0 (a flat surface, c == 0) the backward
    # pass divides by (2A + eps)**2, which underflows to zero in float32 and
    # sends an inf into the where, where it becomes nan. The linear branch below
    # is the one selected for A ~ 0 anyway.
    quadratic = ~is_linear
    two_a = jnp.where(quadratic, 2 * A, 1.0)
    t1 = jnp.where(quadratic, (-B - sqrt_disc) / two_a, jnp.inf)
    t2 = jnp.where(quadratic, (-B + sqrt_disc) / two_a, jnp.inf)

    # Select the nearest forward root that sits on the sag branch. Taking the
    # nearest root outright would return the far sheet of the closed conic for a
    # ray starting outside it (see the note above): the hit would be a surface
    # point the ray never reaches, and since the aperture test only looks at its
    # (x, y) it can land inside an unrelated element and be accepted.
    z1 = oz + t1 * dz
    z2 = oz + t2 * dz
    on_branch1 = (1 + k) * c * z1 <= 1.0 + 1e-9
    on_branch2 = (1 + k) * c * z2 <= 1.0 + 1e-9
    t1_valid = (t1 > t_lo) & on_branch1
    t2_valid = (t2 > t_lo) & on_branch2
    t_conic = jnp.where(
        t1_valid & t2_valid,
        jnp.minimum(t1, t2),
        jnp.where(t1_valid, t1, jnp.where(t2_valid, t2, jnp.inf)),
    )
    t_conic = jnp.where(no_intersection, jnp.inf, t_conic)

    # Select: plane -> linear -> quadratic
    t_conic = jnp.where(is_linear, t_linear, t_conic)
    return t_offset + jnp.where(is_plane, t_plane, t_conic)


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
    # Iterate next to the surface rather than at the source's distance (see
    # _reorigin): g(t) = z - sag(x, y) otherwise differences two numbers of order
    # the source distance to get a sag of order millimetres, and both the
    # residual test and the returned hit coordinates are then noise.
    origin, t_offset = _reorigin(ray_origin, ray_direction, jnp.zeros(3))
    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]

    # Initial guess: use provided value or intersect with z=0 plane. A caller's
    # guess is an absolute t, so bring it into the shifted frame.
    if t_init is None:
        t_init = jnp.where(jnp.abs(dz) > 1e-10, -oz / dz, 0.0)
    else:
        t_init = t_init - t_offset

    def g(t):
        """Implicit function: g(t) = 0 at intersection."""
        x = ox + t * dx
        y = oy + t * dy
        z = oz + t * dz
        return z - sag_fn(x, y)

    def newton_step(carry, _):
        t, converged = carry
        g_val, g_prime = jax.value_and_grad(g)(t)

        # Avoid division by zero
        g_prime_safe = jnp.where(jnp.abs(g_prime) > 1e-12, g_prime, 1e-12)
        t_new = t - g_val / g_prime_safe

        # Check convergence
        new_converged = converged | (jnp.abs(g_val) < tol)

        # Only update if not converged
        t_out = jnp.where(converged, t, t_new)

        return (t_out, new_converged), None

    (t_final, _), _ = jax.lax.scan(newton_step, (t_init, False), None, length=max_iter)

    # Compute hit coordinates
    x_hit = ox + t_final * dx
    y_hit = oy + t_final * dy
    hit_xy = jnp.array([x_hit, y_hit])

    # Check validity: t should be positive and residual small. Forward-ness is a
    # property of the total, not of the shifted-frame parameter.
    residual = jnp.abs(g(t_final))
    valid = (t_offset + t_final > 1e-8) & (residual < tol * 100)

    t_out = jnp.where(valid, t_offset + t_final, jnp.inf)

    return t_out, hit_xy, valid
