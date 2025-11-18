import jax
import jax.numpy as jnp
from ..core.reflection import roughen_normals


class MCIntegrator:
    def __init__(self, n_samples=128, roughness=0.0):
        self.n_samples = n_samples
        self.roughness = roughness

    def sample(self, surfaces, apertures, key):
        N = len(surfaces)
        keys = jax.random.split(key, N+1)
        
        # Sample all mirrors
        mirror_points = []
        mirror_normals = []
        mirror_areas = []
        for i in range(N):
            xy = apertures[i].sample(keys[i], (self.n_samples,))
            m_points, m_normals = surfaces[i].point_and_normal(xy)
            mirror_points.append(m_points)
            mirror_normals.append(m_normals)
            mirror_areas.append(apertures[i].area())

        mirror_points = jnp.array(mirror_points)
        mirror_normals = jnp.array(mirror_normals)
        mirror_areas = jnp.array(mirror_areas)
        
        if self.roughness > 0:
            mirror_normals = roughen_normals(mirror_normals, self.roughness, keys[-1])
            
        mirror_weights = jnp.sum(mirror_normals * jnp.array([0,0,1]), axis=-1, keepdims=True)
        mirror_weights = mirror_weights / mirror_areas[:,None,None] * self.n_samples

        return mirror_points, mirror_normals, mirror_weights