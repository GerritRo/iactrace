from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from . import _salts
from .apertures import Aperture
from .bsdf import BSDF, GaussianBSDF
from .interactions import (
    InteractionType,
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
)
from .surfaces import SurfaceGroup
from .transforms import euler_to_matrix, transform_to_world

InteractionModule = ReflectInteraction | RefractInteraction | SlabInteraction


def _rotate_into_local(rot, vx, vy, vz):
    """rot.T @ v for batched v, written out componentwise for speed."""
    return (
        rot[0, 0] * vx + rot[1, 0] * vy + rot[2, 0] * vz,
        rot[0, 1] * vx + rot[1, 1] * vy + rot[2, 1] * vz,
        rot[0, 2] * vx + rot[1, 2] * vy + rot[2, 2] * vz,
    )


class OpticalElementGroup(eqx.Module):
    """Optical element group composing surface + aperture + interaction.

    All optical elements (mirrors, lenses, slabs) are instances of this class
    configured with appropriate modules.
    """

    # Transform
    positions: Array  # (N, 3)
    rotations: Array  # (N, 3) euler angles in degrees

    # Composable modules
    surface: SurfaceGroup
    aperture: Aperture
    interaction_module: InteractionModule
    bsdf: BSDF

    # PRNG state for sampling and roughness
    sample_key: Array  # PRNGKey

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
    def interaction(self) -> InteractionType:
        return self.interaction_module.interaction_type

    @property
    def kind(self) -> Literal["mirror", "lens", "slab"]:
        """User-facing element kind, derived from the interaction module."""
        return self.interaction_module.kind

    # Geometry

    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates.

        Samples are generated at call time using the stored n_samples and
        sample_key.

        Returns
        -------
        Tuple of (points_world, normals_world, weights) arrays.
        """
        sampling_key = jax.random.fold_in(self.sample_key, _salts.APERTURE)
        aperture_samples = self.aperture.sample(sampling_key, self.n_samples)

        aperture_data = self.aperture.get_area_data()
        area_fn = self.aperture.area_fn
        return transform_to_world(
            aperture_samples,
            self.surface,
            aperture_data,
            self.positions,
            self.rotations,
            area_fn=area_fn,
        )

    def sample_primary_geometry(self, roughness_salt):
        """Sample this group's aperture, with this group's surface roughness applied.

        Parameters
        ----------
        roughness_salt
            Integer folded into this group's sample_key
            to draw the roughness perturbation.

        Returns
        -------
        Tuple of (points_world, normals_world, weights) arrays, as
        transform_to_world, with normals_world perturbed.
        """
        points, normals, weights = self.transform_to_world()
        normals = self.perturb_normals(normals, roughness_salt)
        return points, normals, weights

    # Per-element intersection and interaction

    def intersect_t(self, element_idx, origins, directions):
        """Hit distance only, for the nearest-hit search over a stage.

        Parameters
        ----------
        element_idx
            Index of the element within this group.
        origins, directions : array, shape (n_rays, 3)
            Rays in world coordinates.

        Returns
        -------
        (n_rays,) hit distances, inf where the ray misses the
        surface or lands outside the element's aperture.
        """
        pos = self.positions[element_idx]
        rot = euler_to_matrix(self.rotations[element_idx])

        o_loc = jnp.stack(
            _rotate_into_local(
                rot,
                origins[:, 0] - pos[0],
                origins[:, 1] - pos[1],
                origins[:, 2] - pos[2],
            ),
            axis=-1,
        )
        d_loc = jnp.stack(
            _rotate_into_local(rot, directions[:, 0], directions[:, 1], directions[:, 2]),
            axis=-1,
        )

        t, x, y = jax.vmap(lambda o, d: self.surface.intersect_t_at(element_idx, o, d))(
            o_loc, d_loc
        )
        return jnp.where(self.aperture.check(x, y, element_idx), t, jnp.inf)

    def hit_geometry(self, element_idx, origins, directions):
        """World-frame hit point and normal, for a per-ray element index.

        The counterpart to intersect_t: once the search knows which
        element each ray settled on, this evaluates the surface there.

        Parameters
        ----------
        element_idx : array, shape (n_rays,)
            Per-ray element index.
        origins, directions : array, shape (n_rays, 3)
            Rays in world coordinates.

        Returns
        -------
        Tuple of (points_world, normals_world), each (n_rays, 3).
        """
        pos = self.positions[element_idx]
        rot = jax.vmap(euler_to_matrix)(self.rotations[element_idx])
        rot_t = jnp.swapaxes(rot, 1, 2)

        o_loc = jnp.einsum("nij,nj->ni", rot_t, origins - pos)
        d_loc = jnp.einsum("nij,nj->ni", rot_t, directions)

        _, pts_loc, norms_loc = jax.vmap(self.surface.intersect_at)(element_idx, o_loc, d_loc)

        pts_world = jnp.einsum("nij,nj->ni", rot, pts_loc) + pos
        norms_world = jnp.einsum("nij,nj->ni", rot, norms_loc)
        return pts_world, norms_world

    def perturb_normals(self, normals, roughness_salt, element_idx=None):
        """Apply this group's own BSDF surface-roughness perturbation.

        roughness_salt is folded into this group's sample_key.
        """
        key = jax.random.fold_in(self.sample_key, roughness_salt)
        return self.bsdf.perturb_normals(normals, key, element_idx)

    def apply_interaction(
        self, directions, normals, points, element_idx, current_n, wavelength=None
    ):
        """Apply this group's physical interaction (reflect/refract/slab) at a hit.

        See Interaction.apply for the return value.
        """
        return self.interaction_module.apply(
            directions, normals, points, element_idx, current_n, wavelength
        )

    def interact(
        self, directions, normals, points, element_idx, current_n, roughness_salt, wavelength=None
    ):
        """Perturb normals for roughness, then apply the physical interaction.

        See Interaction.apply for the return value.
        """
        perturbed = self.perturb_normals(normals, roughness_salt, element_idx)
        return self.apply_interaction(
            directions, perturbed, points, element_idx, current_n, wavelength
        )
