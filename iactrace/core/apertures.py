import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

from ..utils.sampling import sample_disk, sample_polygon


class Aperture(eqx.Module):
    """Abstract base class for aperture shapes."""
    
    @abstractmethod
    def sample(self, key, shape):
        """Sample 2D aperture coordinates."""
        pass
    
    @abstractmethod
    def area(self):
        """Return aperture area."""
        pass


class DiskAperture(Aperture):
    """Circular aperture."""
    
    radius: float = eqx.field(static=True)
    
    def __init__(self, radius=1.0):
        self.radius = float(radius)
    
    def sample(self, key, shape):
        pts = sample_disk(key, shape)
        return pts * self.radius

    def area(self):
        return jnp.pi * self.radius**2


class PolygonAperture(Aperture):
    """Convex polygonal aperture."""
    
    vertices: jax.Array
    
    def __init__(self, vertices):
        self.vertices = jnp.asarray(vertices)
    
    def sample(self, key, shape):
        return sample_polygon(key, self.vertices, shape)
    
    def area(self):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
