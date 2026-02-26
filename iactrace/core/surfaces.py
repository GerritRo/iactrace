import equinox as eqx
import jax
import jax.numpy as jnp

from .intersections import intersect_conic, newton_raphson_intersect


def sag_raw(x, y, curvature, conic, aspheric):
    """Compute surface sag z(x,y) without offset.

    Standalone function enabling gradient flow from parameters to sag values.

    Args:
        x: x-coordinate (scalar)
        y: y-coordinate (scalar)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        z: Surface sag at (x, y)
    """
    r2 = x * x + y * y
    c = curvature
    k = conic

    denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
    z = r2 * c / denom

    # Add aspheric terms if any (static check for JAX tracing)
    if aspheric.size > 0:
        powers = jnp.arange(2, 2 + 2 * len(aspheric), 2)
        z = z + jnp.sum(aspheric * r2 ** powers)

    return z


def sag(x, y, offset, curvature, conic, aspheric):
    """Compute surface sag z(x,y) in local mirror coordinates.

    Args:
        x: x-coordinate in local mirror frame (scalar)
        y: y-coordinate in local mirror frame (scalar)
        offset: (x0, y0) offset on parent surface (2,)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        z: Surface sag at (x, y) relative to offset point
    """
    x0, y0 = offset[0], offset[1]
    z0 = sag_raw(x0, y0, curvature, conic, aspheric)
    return sag_raw(x + x0, y + y0, curvature, conic, aspheric) - z0


def compute_sag_and_normal(x, y, offset, curvature, conic, aspheric):
    """Compute surface point and normal at (x, y) with given parameters.

    Args:
        x: x-coordinate in local mirror frame (scalar)
        y: y-coordinate in local mirror frame (scalar)
        offset: (x0, y0) offset on parent surface (2,)
        curvature: Surface curvature (1/radius)
        conic: Conic constant k
        aspheric: Array of aspheric coefficients (K,)

    Returns:
        point: 3D surface point (3,)
        normal: Surface normal (3,), normalized
    """
    z = sag(x, y, offset, curvature, conic, aspheric)
    point = jnp.stack([x, y, z], axis=-1)

    # Compute normal via autodiff
    x_surf = x + offset[0]
    y_surf = y + offset[1]
    dzdx = jax.grad(lambda X: sag_raw(X, y_surf, curvature, conic, aspheric))(x_surf)
    dzdy = jax.grad(lambda Y: sag_raw(x_surf, Y, curvature, conic, aspheric))(y_surf)
    n = jnp.array([-dzdx, -dzdy, 1.0])
    normal = n / jnp.linalg.norm(n)

    return point, normal



class AsphericSurface(eqx.Module):
    """Aspheric surface defined by curvature, conic constant, and polynomial terms.

    When is_pure_conic is True, the intersection uses the exact closed-form
    conic solution and skips Newton-Raphson refinement entirely.
    """

    curvature: jax.Array | float
    conic: jax.Array | float
    aspheric: jax.Array  # (K,)
    is_pure_conic: bool = eqx.field(static=True, default=False)

    def sag(self, x, y, offset):
        """Compute surface sag z(x,y) in local mirror coordinates.

        Convenience method that calls the standalone sag function.

        Args:
            x: x-coordinate in local mirror frame (scalar)
            y: y-coordinate in local mirror frame (scalar)
            offset: (x0, y0) offset on parent surface (2,)

        Returns:
            z: Surface sag at (x, y) relative to offset point
        """
        return sag(x, y, offset, self.curvature, self.conic, self.aspheric)

    def intersect(self, ray_origin, ray_direction, offset, max_iter=10, tol=1e-8):
        """Find ray-surface intersection.

        For pure conics (is_pure_conic=True), uses the exact closed-form solution.
        For aspheric surfaces, uses Newton-Raphson refinement on top of the
        conic initial guess.

        Args:
            ray_origin: Ray origin in local coordinates (3,)
            ray_direction: Ray direction (3,)
            offset: Surface offset (x0, y0) (2,)
            max_iter: Maximum Newton-Raphson iterations (aspheric only)
            tol: Convergence tolerance (aspheric only)

        Returns:
            t: Intersection distance
            point: Intersection point (3,)
            normal: Surface normal at intersection (3,)
        """
        # Translate ray origin to raw surface coordinates for conic intersection
        z0 = sag_raw(offset[0], offset[1], self.curvature, self.conic, self.aspheric)
        ray_origin_raw = jnp.array([
            ray_origin[0] + offset[0],
            ray_origin[1] + offset[1],
            ray_origin[2] + z0
        ])

        # Get closed-form conic intersection
        t = intersect_conic(ray_origin_raw, ray_direction, self.curvature, self.conic)

        if not self.is_pure_conic:
            # Refine with Newton-Raphson for aspheric terms
            def sag_fn(x, y):
                return sag(x, y, offset, self.curvature, self.conic, self.aspheric)

            t, hit_xy, _ = newton_raphson_intersect(
                sag_fn, ray_origin, ray_direction, t, max_iter, tol
            )
            x_hit, y_hit = hit_xy[0], hit_xy[1]
        else:
            x_hit = ray_origin[0] + t * ray_direction[0]
            y_hit = ray_origin[1] + t * ray_direction[1]

        # Compute point and normal using standalone functions
        point, normal = compute_sag_and_normal(
            x_hit, y_hit, offset, self.curvature, self.conic, self.aspheric
        )

        return t, point, normal