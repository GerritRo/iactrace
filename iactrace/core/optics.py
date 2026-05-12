from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .apertures import Aperture
from .bsdf import BSDF, GaussianBSDF
from .interactions import (
    InteractionType,
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
)
from .surfaces import AsphericSurfaceGroup
from .transforms import transform_to_world

InteractionModule = ReflectInteraction | RefractInteraction | SlabInteraction


class OpticalElementGroup(eqx.Module):
    """Optical element group composing surface + aperture + interaction.

    All optical elements (mirrors, lenses, slabs) are instances of this class
    configured with appropriate modules.
    """

    # Transform
    positions: Array           # (N, 3)
    rotations: Array           # (N, 3) euler angles in degrees

    # Composable modules
    surface: AsphericSurfaceGroup
    aperture: Aperture
    interaction_module: InteractionModule
    bsdf: BSDF

    # PRNG state for sampling and roughness
    sample_key: Array          # PRNGKey

    n_samples: int = eqx.field(static=True)
    optical_stage: int = eqx.field(static=True)

    def __init__(
        self,
        positions,
        rotations,
        surface,
        aperture,
        interaction_module,
        sample_key,
        optical_stage=0,
        n_samples=100,
        bsdf=None,
    ):
        n_elements = jnp.asarray(positions).shape[0]

        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.surface = surface
        self.aperture = aperture
        self.interaction_module = interaction_module
        self.optical_stage = int(optical_stage)
        self.n_samples = int(n_samples)

        if bsdf is None:
            bsdf = GaussianBSDF(scale=jnp.zeros(n_elements))
        self.bsdf = bsdf

        self.sample_key = sample_key

    def __len__(self):
        return self.positions.shape[0]

    @property
    def n_elements(self) -> int:
        return self.positions.shape[0]

    @property
    def interaction(self) -> InteractionType:
        return self.interaction_module.interaction_type

    @property
    def kind(self) -> Literal["mirror", "lens", "slab"]:
        """User-facing element kind, derived from the interaction module."""
        match self.interaction_module.interaction_type:
            case InteractionType.REFLECT:
                return "mirror"
            case InteractionType.REFRACT:
                return "lens"
            case InteractionType.SLAB:
                return "slab"

    def check_aperture(self, x, y, element_idx):
        return self.aperture.check(x, y, element_idx)

    # Geometry

    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates.

        Samples are generated at call time using the stored n_samples and
        sample_key.

        Returns:
            Tuple of (points_world, normals_world, weights) arrays.
        """
        sampling_key = jax.random.fold_in(self.sample_key, 0x5A3B1E)
        aperture_samples = self.aperture.sample(sampling_key, self.n_samples)

        aperture_data = self.aperture.get_area_data()
        area_fn = self.aperture.area_fn
        return transform_to_world(
            aperture_samples, self.surface, aperture_data,
            self.positions, self.rotations, area_fn=area_fn,
        )
