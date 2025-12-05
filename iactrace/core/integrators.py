import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod

from ..utils.sampling import sample_disk, sample_polygon


class Integrator(ABC):
    """Abstract base class for mirror sampling integrators.

    Different integrators implement different sampling strategies:
    - MCIntegrator: Monte Carlo random sampling
    - Future: GridIntegrator, StratifiedIntegrator, etc.

    Integrators are responsible for ALL sampling logic including:
    - Generating 2D aperture sample points
    - Mapping to 3D surface points
    - Computing surface normals
    - Computing importance weights
    """

    @abstractmethod
    def sample_group(self, group, key):
        """
        Sample a single mirror group and return updated MirrorGroup object.

        Args:
            group: MirrorGroup object
            key: JAX random key

        Returns:
            MirrorGroup with sampled points/normals/weights
        """
        pass

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
        sampled = [self.sample_group(g, k) for g, k in zip(mirror_groups, keys[:-1])]
        return sampled


class MCIntegrator(Integrator):
    """Monte Carlo integrator for mirror groups.

    Uses uniform random sampling over apertures to generate sample points
    on mirror surfaces. Appropriate for most ray tracing applications.

    This integrator owns all sampling logic:
    - For disk apertures: uniform random sampling in polar coordinates
    - For polygon apertures: uniform random sampling via fan triangulation
    """

    def __init__(self, n_samples=128):
        self.n_samples = n_samples

    def sample_group(self, group, key):
        """
        Sample a single mirror group using Monte Carlo sampling.

        The sampling process:
        1. Generate uniform random 2D points on each aperture
        2. Map to 3D surface points using the surface equation
        3. Compute surface normals via automatic differentiation
        4. Compute importance weights: cos(theta) / area * n_samples

        Args:
            group: MirrorGroup object (AsphericDiskMirrorGroup or AsphericPolygonMirrorGroup)
            key: JAX random key

        Returns:
            MirrorGroup with sampled points, normals, and weights
        """
        # Detect group type and delegate to appropriate sampling method
        if hasattr(group, 'radii'):
            return self._sample_disk_group(group, key)
        elif hasattr(group, 'vertices'):
            return self._sample_polygon_group(group, key)
        else:
            raise TypeError(f"Unknown MirrorGroup type: {type(group)}")

    def _sample_disk_group(self, group, key):
        """Sample a group of mirrors with circular disk apertures."""
        n_mirrors = len(group)
        n_samples = self.n_samples
        surface = group.get_surface()

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, radius, offset):
            # Generate uniform random samples in unit disk, scale by radius
            pts = sample_disk(key, (n_samples,))
            xy = pts * radius

            # Map 2D to 3D surface points and normals
            points, normals = surface.point_and_normal(xy, offset)

            # Compute weights: cos(angle to z-axis) / area * n_samples
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            area = jnp.pi * radius**2
            weights = cos_z / area * n_samples

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(
            keys, group.radii, group.offsets
        )

        return eqx.tree_at(
            lambda g: (g.points, g.normals, g.weights),
            group,
            (points, normals, weights)
        )

    def _sample_polygon_group(self, group, key):
        """Sample a group of mirrors with convex polygon apertures."""
        n_mirrors = len(group)
        n_samples = self.n_samples
        surface = group.get_surface()

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, vertices, offset):
            # Generate uniform random samples within polygon
            xy = sample_polygon(key, vertices, (n_samples,))

            # Map 2D to 3D surface points and normals
            points, normals = surface.point_and_normal(xy, offset)

            # Compute weights: cos(angle to z-axis) / area * n_samples
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            # Polygon area using shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
            weights = cos_z / area * n_samples

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(
            keys, group.vertices, group.offsets
        )

        return eqx.tree_at(
            lambda g: (g.points, g.normals, g.weights),
            group,
            (points, normals, weights)
        )