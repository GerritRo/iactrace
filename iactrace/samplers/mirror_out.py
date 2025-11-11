"""Monte Carlo integrator using random disk sampling."""

import jax
import jax.numpy as jnp
from ..utils.sampling import sample_disk
from ..core.surfaces import spherical_surface, parabolic_surface
from ..core.reflection import roughen_normals


class MCIntegrator:
    """
    Monte Carlo integrator for mirror sampling.

    Samples points uniformly on mirror facets and optionally adds surface roughness.
    """

    def __init__(self, n_samples=128, roughness=0.0):
        """
        Args:
            n_samples: Number of sample points per mirror facet
            roughness: RMS surface roughness in arcseconds (0 = perfect mirror)
        """
        self.n_samples = n_samples
        self.roughness = roughness

    def sample_mirrors(self, telescope, key):
        """
        Generate sample points on mirror facets.

        Args:
            telescope: Telescope object
            key: JAX random key

        Returns:
            MirrorArray with sampled points and normals in local coordinates
        """
        n_mirrors = len(telescope.mirror_positions)
        mirror_radii = telescope.mirror_positions[:, 2] if telescope.mirror_positions.shape[1] > 2 else \
                      jnp.full(n_mirrors, 0.6)  # Default mirror radius

        # Sample disk points
        key1, key2 = jax.random.split(key)
        xy = sample_disk(key1, (n_mirrors, self.n_samples)) * mirror_radii[:, None, None] / 2

        # Apply surface function
        if telescope.surface_type == 'spherical':
            m_points, m_normals = spherical_surface(telescope.dish_radius, xy)
        elif telescope.surface_type == 'parabolic':
            m_points, m_normals = parabolic_surface(telescope.focal_length, xy)
        else:
            raise ValueError(f"Unknown surface type: {telescope.surface_type}")

        # Apply roughness if specified
        if self.roughness > 0:
            m_normals = roughen_normals(m_normals, self.roughness, key2)

        return MirrorArray(points=m_points, normals=m_normals)


class GridIntegrator:
    """
    Grid-based integrator for deterministic sampling.

    Uses a uniform grid of points on each mirror facet.
    """

    def __init__(self, n_samples_per_axis=16, roughness=0.0):
        """
        Args:
            n_samples_per_axis: Number of samples along each axis (total = ~n^2)
            roughness: RMS surface roughness in arcseconds
        """
        self.n_samples_per_axis = n_samples_per_axis
        self.roughness = roughness

    def sample_mirrors(self, telescope, key):
        """
        Generate grid sample points on mirror facets.

        Args:
            telescope: Telescope object
            key: JAX random key (used only for roughness)

        Returns:
            MirrorArray with sampled points and normals
        """
        from ..utils.sampling import sample_grid

        n_mirrors = len(telescope.mirror_positions)
        mirror_radii = telescope.mirror_positions[:, 2] if telescope.mirror_positions.shape[1] > 2 else \
                      jnp.full(n_mirrors, 0.6)

        # Generate grid points
        grid_points = sample_grid(self.n_samples_per_axis)
        n_samples = len(grid_points)

        # Replicate for each mirror and scale
        xy = jnp.tile(grid_points[None, :, :], (n_mirrors, 1, 1)) * mirror_radii[:, None, None] / 2

        # Apply surface function
        if telescope.surface_type == 'spherical':
            m_points, m_normals = spherical_surface(telescope.dish_radius, xy)
        elif telescope.surface_type == 'parabolic':
            m_points, m_normals = parabolic_surface(telescope.focal_length, xy)
        else:
            raise ValueError(f"Unknown surface type: {telescope.surface_type}")

        # Apply roughness if specified
        if self.roughness > 0:
            m_normals = roughen_normals(m_normals, self.roughness, key)

        return MirrorArray(points=m_points, normals=m_normals)


class MirrorArray:
    """Container for sampled mirror points and normals."""

    def __init__(self, points, normals):
        """
        Args:
            points: Sample points in local coordinates (n_mirrors, n_samples, 3)
            normals: Surface normals (n_mirrors, n_samples, 3)
        """
        self.points = points
        self.normals = normals
