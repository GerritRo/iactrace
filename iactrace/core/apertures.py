import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod


class Aperture(eqx.Module):
    """Abstract base class for aperture shapes."""

    @abstractmethod
    def area(self):
        """Return aperture area."""
        pass

    @abstractmethod
    def check_aperture(self, x, y):
        """Check if point lies within aperture"""
        pass
    


class DiskAperture(Aperture):
    """Circular aperture."""

    radius: float = eqx.field(static=True)

    def __init__(self, radius=1.0):
        self.radius = float(radius)

    def area(self):
        return jnp.pi * self.radius**2
    
    def check_aperture(self, x, y):
        return x**2 + y**2 <= self.radius**2


class PolygonAperture(Aperture):
    """Convex polygonal aperture."""

    vertices: jax.Array
    n_vertices: int

    def __init__(self, vertices):
        self.vertices = jnp.asarray(vertices)
        self.n_vertices = len(vertices)

    def area(self):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
    
    def check_aperture(self, x, y):
        """Check if points (x, y) are within convex polygon."""
        verts = self.vertices
        n = self.n_vertices
        
        def edge_check(carry, i):
            v1, v2 = verts[i], verts[(i + 1) % n]
            cross = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0])
            return carry & (cross >= 0), None
        
        inside, _ = jax.lax.scan(edge_check, jnp.ones_like(x, dtype=bool), jnp.arange(n))
        return inside