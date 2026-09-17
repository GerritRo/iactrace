from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array

_ARCSEC_TO_RAD = jnp.pi / (180.0 * 3600.0)


# Shared Peturbation helper


def _apply_perturbation(normals, angles, scale):
    """Tilt unit normals by angles (radians, scaled) in their tangent plane.

    Builds an orthonormal tangent frame from each normal, adds the scaled
    offsets, and re-normalises. Shapes broadcast: normals (..., 3), angles
    (..., 2), scale to (..., 1).
    """
    theta1 = angles[..., 0]
    theta2 = angles[..., 1]

    ref_z = jnp.array([0.0, 0.0, 1.0])
    ref_x = jnp.array([1.0, 0.0, 0.0])
    dot_z = jnp.abs(jnp.sum(normals * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)

    tangent1 = jnp.cross(normals, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(normals, tangent1)

    delta = theta1[..., None] * tangent1 + theta2[..., None] * tangent2
    perturbed = normals + scale * delta
    return perturbed / jnp.linalg.norm(perturbed, axis=-1, keepdims=True)


# Base BSDF class


class BSDF(eqx.Module):
    """Abstract base for surface scattering models.

    Subclasses implement _sample_perturbation, returning (angles, scale); the
    base class builds the tangent frame and applies the perturbation.
    """

    @staticmethod
    def _gather(param, element_idx):
        """Resolve a per-element parameter for whichever indexing mode is in play.

        Per-ray (element_idx given) yields (n_rays,); batch mode
        (element_idx is None) yields (N, 1), broadcasting over samples.
        """
        if element_idx is not None:
            return param[element_idx]
        return param[:, None]

    @abc.abstractmethod
    def _sample_perturbation(self, key, shape, element_idx):
        """Draw (angles, scale): (*shape, 2) angle pairs and a radian scale.

        shape is normals.shape[:-1]; element_idx goes to _gather.
        """
        ...

    def perturb_normals(self, normals, key, element_idx=None):
        """Perturb (..., 3) surface normals.

        Works for any leading shape: (n_rays, 3) with a per-ray element_idx,
        or (N, S, 3) with element_idx=None when the element dimension is
        already present.
        """
        angles, scale = self._sample_perturbation(key, normals.shape[:-1], element_idx)
        return _apply_perturbation(normals, angles, scale)


# Gaussian BSDF


class GaussianBSDF(BSDF):
    """Single-Gaussian surface roughness model.

    Perturbs surface normals by Gaussian-distributed random angles.
    This is the standard model for surface microroughness.

    Attributes
    ----------
    scale : array, shape (N,)
        Per-element roughness sigma in arcseconds.
        Zero means perfect specular (no perturbation).
    """

    scale: Array  # (N,) in arcseconds

    def _sample_perturbation(self, key, shape, element_idx):
        angles = jr.normal(key, (*shape, 2))
        scale = self._gather(self.scale, element_idx)[..., None] * _ARCSEC_TO_RAD
        return angles, scale


# Double-Gaussian BSDF


class DoubleGaussianBSDF(BSDF):
    """Mixture of two Gaussians for surfaces with multi-scale roughness.

    Models surfaces that have both fine-scale microroughness (narrow
    component) and broader scattering from mid-spatial-frequency errors
    (wide component). Each ray's perturbation is drawn from the narrow
    Gaussian with probability (1 - mix_weight) or from the wide
    Gaussian with probability mix_weight.

    Attributes
    ----------
    scale_narrow : array, shape (N,)
        Per-element narrow-component sigma in arcseconds.
    scale_wide : array, shape (N,)
        Per-element wide-component sigma in arcseconds.
    mix_weight : array, shape (N,)
        Per-element probability of the wide component, in [0, 1].
    """

    scale_narrow: Array  # (N,) in arcseconds
    scale_wide: Array  # (N,) in arcseconds
    mix_weight: Array  # (N,)

    def _sample_perturbation(self, key, shape, element_idx):
        k_angles, k_select = jr.split(key)
        angles = jr.normal(k_angles, (*shape, 2))

        wide = jr.uniform(k_select, shape) < self._gather(self.mix_weight, element_idx)
        sn = self._gather(self.scale_narrow, element_idx)
        sw = self._gather(self.scale_wide, element_idx)
        scale = jnp.where(wide, sw, sn)[..., None] * _ARCSEC_TO_RAD

        return angles, scale
