from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import random

from ..utils.sampling import sample_disk, sample_polygon

class Aperture(ABC):
    """Abstract base class for aperture shapes."""
    
    @abstractmethod
    def sample(self, key, shape):
        """
        Sample 2D aperture coordinates.
        Args:
            key : PRNGKey
            shape : tuple, batch shape (e.g. (n_mirrors, n_samples))
        Returns:
            (..., 2) array
        """
        pass
    

class DiskAperture(Aperture):
    def __init__(self, radius=1.0):
        self.radius = radius
    
    def sample(self, key, shape):
        pts = sample_disk(key, shape)
        return pts * self.radius

    
class PolygonAperture(Aperture):
    def __init__(self, vertices):
        """
        vertices: (n,2) jnp.array, convex polygon
        """
        self.vertices = jnp.asarray(vertices)

    def sample(self, key, shape):
        return sample_polygon(key, self.vertices, shape)
