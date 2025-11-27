import jax
import jax.numpy as jnp
import equinox as eqx

from ..core.reflection import roughen_normals


class MCIntegrator:
    """Monte Carlo integrator for sampling mirror surfaces."""
    
    def __init__(self, n_samples=128, roughness=0.0):
        self.n_samples = n_samples
        self.roughness = roughness
    
    def sample_mirrors(self, mirrors, key):
        """
        Sample all mirrors and return list of updated Mirror objects.
        
        Args:
            mirrors: List of Mirror objects
            key: JAX random key
        
        Returns:
            List of Mirror objects with sampled points/normals/weights
        """
        keys = jax.random.split(key, len(mirrors) + 1)
        
        # Sample each mirror
        sampled = [m.sample(self.n_samples, k) for m, k in zip(mirrors, keys[:-1])]
        
        # Apply roughness if needed
        if self.roughness > 0:
            sampled = self._apply_roughness(sampled, keys[-1])
        
        return sampled
    
    def _apply_roughness(self, mirrors, key):
        """Apply surface roughness to all mirrors."""        
        # Stack normals for efficient roughening
        normals_stacked = jnp.stack([m.normals for m in mirrors])  # (N, M, 3)
        roughened = roughen_normals(normals_stacked, self.roughness, key)
        
        # Update each mirror with roughened normals
        return [
            eqx.tree_at(lambda m: m.normals, mirror, roughened[i])
            for i, mirror in enumerate(mirrors)
        ]
