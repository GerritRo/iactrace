"""Surface generation functions for mirror facets."""
import jax
import jax.numpy as jnp
import equinox as eqx


class AsphericSurface(eqx.Module):
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

        # Base conic sag
        denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
        z = r2 * c / denom

        # Aspheric polynomial (even powers: r^4, r^6, ...)
        if self.aspheric.size > 0:
            powers = jnp.arange(2, 2 + 2 * len(self.aspheric), 2)
            z += jnp.sum(self.aspheric * r2 ** powers)

        return z


    def point(self, x, y):
        """Return 3D surface point."""
        return jnp.stack([x, y, self.sag(x, y)], axis=-1)


    def normal(self, x, y):
        """Return surface normal using JAX autodiff."""
        sag_fn = lambda x, y: self.sag(x, y)

        dzdx = jax.grad(lambda X: sag_fn(X, y))(x)
        dzdy = jax.grad(lambda Y: sag_fn(x, Y))(y)

        n = jnp.array([-dzdx, -dzdy, 1.0])
        return n / jnp.linalg.norm(n)
    
    def point_and_normal(self, xy):
        x,y  = xy[...,0], xy[...,1]
        points = jax.vmap(self.point, in_axes=(0, 0))(x, y)
        normals = jax.vmap(self.normal, in_axes=(0, 0))(x, y)
        return points, normals