import jax
import jax.numpy as jnp
import equinox as eqx

from .reflection import roughen_normals


class MCIntegrator:
    """Monte Carlo integrator for mirror groups."""

    def __init__(self, n_samples=128):
        self.n_samples = n_samples

    def sample_mirror_groups(self, mirror_groups, key):
        """
        Sample all mirror groups and return list of updated MirrorGroup objects.

        Args:
            mirror_groups: List of MirrorGroup objects
            key: JAX random key

        Returns:
            List of MirrorGroup objects with sampled points/normals/weights
        """
        if not mirror_groups:
            return []

        keys = jax.random.split(key, len(mirror_groups) + 1)

        # Sample each group
        sampled = [g.sample(self.n_samples, k) for g, k in zip(mirror_groups, keys[:-1])]

        return sampled