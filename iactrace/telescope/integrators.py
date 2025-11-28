import jax
import jax.numpy as jnp
import equinox as eqx

from ..core.reflection import roughen_normals


class MCIntegrator:
    """Monte Carlo integrator for mirror groups."""

    def __init__(self, n_samples=128, roughness=0.0):
        self.n_samples = n_samples
        self.roughness = roughness

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

        # Apply roughness if needed
        if self.roughness > 0:
            sampled = self._apply_roughness_to_groups(sampled, keys[-1])

        return sampled

    def _apply_roughness_to_groups(self, mirror_groups, key):
        """Apply surface roughness to all mirror groups."""
        # Process each group independently
        result = []
        n_groups = len(mirror_groups)
        keys = jax.random.split(key, n_groups)

        for group, k in zip(mirror_groups, keys):
            # group.normals has shape (N_mirrors, M_samples, 3)
            roughened = roughen_normals(group.normals, self.roughness, k)
            updated_group = eqx.tree_at(lambda g: g.normals, group, roughened)
            result.append(updated_group)

        return result