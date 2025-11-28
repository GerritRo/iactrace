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

    def sag(self, x, y):
        """Return surface sag z(x,y)."""
        r2 = x * x + y * y
        c = self.curvature
        k = self.conic

        denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
        z = r2 * c / denom

        if self.aspheric.size > 0:
            powers = jnp.arange(2, 2 + 2 * len(self.aspheric), 2)
            z += jnp.sum(self.aspheric * r2 ** powers)

        return z

    def point(self, x, y):
        """Return 3D surface point."""
        return jnp.stack([x, y, self.sag(x, y)], axis=-1)

    def normal(self, x, y):
        """Return surface normal using autodiff."""
        dzdx = jax.grad(lambda X: self.sag(X, y))(x)
        dzdy = jax.grad(lambda Y: self.sag(x, Y))(y)
        n = jnp.array([-dzdx, -dzdy, 1.0])
        return n / jnp.linalg.norm(n)
    
    def point_and_normal(self, xy):
        """Return points and normals for batch of (x, y) coordinates."""
        x, y = xy[..., 0], xy[..., 1]
        points = jax.vmap(self.point)(x, y)
        normals = jax.vmap(self.normal)(x, y)
        return points, normals
