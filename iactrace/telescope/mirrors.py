import jax
import jax.numpy as jnp
import equinox as eqx

from ..core.transforms import euler_to_matrix

class Mirror(eqx.Module):
    """Single mirror element with surface, aperture, and sampled geometry."""
    
    position: jax.Array      # (3,)
    rotation: jax.Array      # (3,) euler angles in degrees
    surface: eqx.Module      # AsphericSurface
    aperture: eqx.Module     # DiskAperture, PolygonAperture
    
    # Sampled geometry in local coordinates
    points: jax.Array        # (M, 3)
    normals: jax.Array       # (M, 3)
    weights: jax.Array       # (M, 1)
    
    def __init__(self, position, rotation, surface, aperture,
                 points=None, normals=None, weights=None):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.surface = surface
        self.aperture = aperture
        self.points = points if points is not None else jnp.zeros((0, 3))
        self.normals = normals if normals is not None else jnp.zeros((0, 3))
        self.weights = weights if weights is not None else jnp.zeros((0, 1))
    
    def sample(self, n_samples, key):
        """Return new Mirror with sampled surface points and normals."""
        xy = self.aperture.sample(key, (n_samples,))
        points, normals = self.surface.point_and_normal(xy)
        
        # Weight = cos(angle to z-axis) normalized by area
        cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
        weights = cos_z / self.aperture.area() * n_samples
        
        return eqx.tree_at(
            lambda m: (m.points, m.normals, m.weights),
            self,
            (points, normals, weights)
        )
    
    def transform_to_world(self):
        """Return world-space points, normals, and weights."""
        rot = euler_to_matrix(self.rotation)
        points_world = jnp.einsum('ij,nj->ni', rot, self.points) + self.position
        normals_world = jnp.einsum('ij,nj->ni', rot, self.normals)
        return points_world, normals_world, self.weights
