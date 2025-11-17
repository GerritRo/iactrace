import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from ..utils.sampling import sample_disk, sample_polygon


class Aperture(eqx.Module):
    """Abstract base class for aperture shapes."""
    
    @abstractmethod
    def sample(self, key, shape):
        """
        Sample 2D aperture coordinates.
        
        Args:
            key: PRNGKey
            shape: tuple, batch shape (e.g. (n_mirrors, n_samples))
        
        Returns:
            (..., 2) array
        """
        pass


class DiskAperture(Aperture):
    """Circular aperture."""
    
    radius: float = eqx.field(static=True)
    
    def __init__(self, radius=1.0):
        self.radius = float(radius)
    
    def sample(self, key, shape):
        pts = sample_disk(key, shape)
        return pts * self.radius


class PolygonAperture(Aperture):
    """Polygonal aperture."""
    
    vertices: jax.Array
    
    def __init__(self, vertices):
        """
        Args:
            vertices: (n, 2) array, convex polygon vertices
        """
        self.vertices = jnp.asarray(vertices)
    
    def sample(self, key, shape):
        return sample_polygon(key, self.vertices, shape)