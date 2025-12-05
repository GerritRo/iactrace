import jax
import jax.numpy as jnp
import equinox as eqx

from .intersections import intersect_conic, newton_raphson_intersect


class AsphericSurface(eqx.Module):
    """Aspheric surface defined by curvature, conic constant, and polynomial terms."""
    
    curvature: float
    conic: float
    aspheric: jax.Array  # (K,)

    @staticmethod
    def from_template(tmpl):
        """Convert dict from YAML into AsphericSurface."""
        s = tmpl["surface"]
        return AsphericSurface(
            curvature=float(s["curvature"]),
            conic=float(s["conic"]),
            aspheric=jnp.array(s.get("aspheric", []), dtype=jnp.float32),
        )

    def _sag_raw(self, x, y):
        """Return surface sag z(x,y) without offset."""
        r2 = x * x + y * y
        c = self.curvature
        k = self.conic

        denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
        z = r2 * c / denom

        # Static check - okay for JAX since aspheric.size is known at trace time
        if self.aspheric.size > 0:
            powers = jnp.arange(2, 2 + 2 * len(self.aspheric), 2)
            z += jnp.sum(self.aspheric * r2 ** powers)

        return z

    def sag(self, x, y, offset):
        """Return surface sag z(x,y) in local mirror coordinates."""
        x0, y0 = offset[0], offset[1]
        z0 = self._sag_raw(x0, y0)
        return self._sag_raw(x + x0, y + y0) - z0

    def point(self, x, y, offset):
        """Return 3D surface point in local mirror coordinates."""
        return jnp.stack([x, y, self.sag(x, y, offset)], axis=-1)

    def normal(self, x, y, offset):
        """Return surface normal using autodiff."""
        x_surf = x + offset[0]
        y_surf = y + offset[1]
        dzdx = jax.grad(lambda X: self._sag_raw(X, y_surf))(x_surf)
        dzdy = jax.grad(lambda Y: self._sag_raw(x_surf, Y))(y_surf)
        n = jnp.array([-dzdx, -dzdy, 1.0])
        return n / jnp.linalg.norm(n)
    
    def point_and_normal(self, xy, offset):
        """Return points and normals for batch of (x, y) coordinates."""
        x, y = xy[..., 0], xy[..., 1]
        points = jax.vmap(lambda xi, yi: self.point(xi, yi, offset))(x, y)
        normals = jax.vmap(lambda xi, yi: self.normal(xi, yi, offset))(x, y)
        return points, normals

    def intersect(self, ray_origin, ray_direction, offset, max_iter=10, tol=1e-8):
        """
        Find ray-surface intersection using Newton-Raphson iteration.

        Uses closed-form conic intersection as initial guess, then refines
        with Newton-Raphson to account for aspheric terms.

        Args:
            ray_origin: Ray origin in local coordinates (3,)
            ray_direction: Ray direction in local coordinates (3,), assumed normalized
            offset: (x0, y0) offset on parent surface (2,)
            max_iter: Maximum Newton-Raphson iterations
            tol: Convergence tolerance

        Returns:
            t: Parameter along ray (scalar), inf if no intersection
            point: Intersection point (3,)
            normal: Surface normal at intersection (3,)
        """
        # Translate ray origin to raw surface coordinates for conic intersection
        z0 = self._sag_raw(offset[0], offset[1])
        ray_origin_raw = jnp.array([
            ray_origin[0] + offset[0],
            ray_origin[1] + offset[1],
            ray_origin[2] + z0
        ])

        # Get initial guess from closed-form conic intersection
        t_init = intersect_conic(ray_origin_raw, ray_direction, self.curvature, self.conic)

        # Refine with Newton-Raphson
        sag_fn = lambda x, y: self.sag(x, y, offset)

        t, hit_xy, _ = newton_raphson_intersect(
            sag_fn, ray_origin, ray_direction, t_init, max_iter, tol
        )

        x_hit, y_hit = hit_xy[0], hit_xy[1]
        point = self.point(x_hit, y_hit, offset)
        normal = self.normal(x_hit, y_hit, offset)

        return t, point, normal