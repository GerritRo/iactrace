import jax
import jax.numpy as jnp
import equinox as eqx


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
        ox, oy, oz = ray_origin[0], ray_origin[1], ray_origin[2]
        dx, dy, dz = ray_direction[0], ray_direction[1], ray_direction[2]
        
        # Initial guess: intersect with z=0 plane
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
            return z - self.sag(x, y, offset)
        
        g_grad = jax.grad(g)
        
        def newton_step(carry, _):
            t, converged = carry
            g_val = g(t)
            g_prime = g_grad(t)
            
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
        
        # Compute intersection point and normal
        x_hit = ox + t_final * dx
        y_hit = oy + t_final * dy
        point = self.point(x_hit, y_hit, offset)
        normal = self.normal(x_hit, y_hit, offset)
        
        # Check validity: t should be positive and residual small
        residual = jnp.abs(g(t_final))
        valid = (t_final > 1e-8) & (residual < tol * 100)
        
        t_out = jnp.where(valid, t_final, jnp.inf)
        
        return t_out, point, normal