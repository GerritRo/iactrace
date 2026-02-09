from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..utils.sampling import sample_annulus, sample_polygon
from .optics import generate_perturbation_angles

if TYPE_CHECKING:
    from ..telescope.optical_base import OpticalGroupBase


class Integrator(ABC):
    """
    Abstract base class for optical element sampling integrators.
    """

    @abstractmethod
    def sample_group(self, group: OpticalGroupBase, key: Array) -> OpticalGroupBase:
        """
        Sample a single optical group and return updated object.

        Args:
            group: OpticalGroupBase object (mirror or lens)
            key: JAX random key

        Returns:
            OpticalGroup with sampled aperture positions and perturbation angles
        """
        ...


    def sample_optical_groups(
        self, optical_groups: list[OpticalGroupBase], key: Array
    ) -> list[OpticalGroupBase]:
        """
        Sample all optical groups (mirrors and lenses) and return updated list.

        This is the unified entry point for sampling any optical element.

        Args:
            optical_groups: List of OpticalGroupBase objects
            key: JAX random key

        Returns:
            List of OpticalGroupBase objects with sampled aperture positions
        """
        if not optical_groups:
            return []

        keys = jax.random.split(key, len(optical_groups) + 1)
        sampled = [self.sample_group(g, k) for g, k in zip(optical_groups, keys[:-1], strict=False)]
        return sampled


class MCIntegrator(Integrator):
    """
    Monte Carlo integrator for optical element groups (mirrors and lenses).
    """

    n_samples: int

    def __init__(self, n_samples: int = 128) -> None:
        self.n_samples = n_samples

    def sample_group(self, group: OpticalGroupBase, key: Array) -> OpticalGroupBase:
        """
        Sample a single optical group using Monte Carlo sampling.

        Works with both mirror groups (AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup)
        and lens groups (AsphericDiskLensGroup, PlanoSlabGroup).

        The sampling process:
        1. Generate uniform random 2D points on each aperture
        2. Generate random perturbation angles for surface roughness
        3. Geometry (3D points, normals, weights) is computed at render time

        Args:
            group: OpticalGroupBase object (mirror or lens)
            key: JAX random key

        Returns:
            OpticalGroup with sampled aperture positions and perturbation angles
        """
        params = group.get_sampling_params()
        group_type = params["type"]

        n_elements = len(group)
        n_samples = self.n_samples
        keys = jax.random.split(key, n_elements)

        if group_type == "disk":
            radii = params["radii"]
            inner_radii = params["inner_radii"]
            aperture_samples, angles = jax.vmap(
                lambda mkey, inner_r, outer_r: self._sample_single_disk(
                    mkey, inner_r, outer_r, n_samples
                )
            )(keys, inner_radii, radii)
        elif group_type == "polygon":
            vertices = params["vertices"]
            aperture_samples, angles = jax.vmap(
                lambda mkey, verts: self._sample_single_polygon(mkey, verts, n_samples)
            )(keys, vertices)
        else:
            raise TypeError(f"Unknown aperture type: {group_type}")

        # Use remaining key for perturbation_key (for ray tracing roughness)
        perturbation_key = jax.random.fold_in(key, 0xDEADBEEF)

        return eqx.tree_at(
            lambda g: (g.aperture_samples, g.perturbation_angles, g.perturbation_key),
            group,
            (aperture_samples, angles, perturbation_key),
        )

    def _sample_single_disk(self, key: Array, inner_radius: Array,
                             outer_radius: Array, n_samples: int):
        """Sample a single disk/annular aperture.

        Uses annulus sampling which generalizes to solid disk when inner_radius=0.

        Args:
            key: JAX random key
            inner_radius: Inner radius (0 for solid disk, >0 for center hole)
            outer_radius: Outer radius
            n_samples: Number of samples

        Returns:
            Tuple of (xy_samples, perturbation_angles)
        """
        key_sample, key_perturb = jax.random.split(key)

        # Generate uniform random samples in annulus (works for disk when inner=0)
        xy = sample_annulus(key_sample, inner_radius, outer_radius, (n_samples,))

        # Generate random perturbation angles
        angles = self._generate_angles(key_perturb, n_samples)

        return xy, angles

    def _sample_single_polygon(self, key: Array, vertices: Array, n_samples: int):
        """Sample a single polygon aperture.

        Args:
            key: JAX random key
            vertices: Polygon vertices (K, 2)
            n_samples: Number of samples

        Returns:
            Tuple of (xy_samples, perturbation_angles)
        """
        key_sample, key_perturb = jax.random.split(key)

        # Generate uniform random samples within polygon
        xy = sample_polygon(key_sample, vertices, (n_samples,))

        # Generate random perturbation angles
        angles = self._generate_angles(key_perturb, n_samples)

        return xy, angles

    def _generate_angles(self, key: Array, n_samples: int):
        """Generate random perturbation angles.

        Args:
            key: JAX random key
            n_samples: Number of samples

        Returns:
            Random angles (n_samples, 2)
        """
        dummy_normals = jnp.zeros((n_samples, 3))
        return generate_perturbation_angles(dummy_normals, key)
