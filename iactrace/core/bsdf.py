import abc

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array

# Conversion factor: arcseconds -> radians
_ARCSEC_TO_RAD = jnp.pi / (180.0 * 3600.0)


# Shared perturbation helper

def _apply_perturbation(normals, angles, scale):
    """Perturb normals by random angles along a tangent basis.

    Builds an orthonormal tangent frame from the normals, applies
    scaled random offsets, and re-normalizes.

    Args:
        normals: Surface normals (..., 3), unit length.
        angles: Random angle pairs (..., 2).
        scale: Per-element perturbation scale, broadcastable to (..., 1).

    Returns:
        Perturbed normals (..., 3), unit length.
    """
    theta1 = angles[..., 0]
    theta2 = angles[..., 1]

    ref_z = jnp.array([0., 0., 1.])
    ref_x = jnp.array([1., 0., 0.])
    dot_z = jnp.abs(jnp.sum(normals * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)

    tangent1 = jnp.cross(normals, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(normals, tangent1)

    delta = theta1[..., None] * tangent1 + theta2[..., None] * tangent2
    perturbed = normals + scale * delta
    return perturbed / jnp.linalg.norm(perturbed, axis=-1, keepdims=True)


# Base class

class BSDF(eqx.Module):
    """Abstract base for surface scattering models.

    Subclasses implement :meth:`_sample_perturbation` which returns
    ``(angles, scale)`` for a given shape and element-index resolver.
    The base class handles tangent-frame construction and applies the
    perturbation.
    """

    @staticmethod
    def _gather(param, element_idx):
        """Resolve a per-element parameter for the current indexing mode.

        Per-ray mode (element_idx provided):
            ``param[element_idx]`` → ``(n_rays,)``.
        Batch mode (element_idx is None):
            ``param[:, None]`` → ``(N, 1)``, broadcasts over samples.

        Subclasses call this inside ``_sample_perturbation`` instead of
        writing their own conditional indexing logic.
        """
        if element_idx is not None:
            return param[element_idx]
        return param[:, None]

    @abc.abstractmethod
    def _sample_perturbation(self, key, shape, element_idx):
        """Draw perturbation angles and per-ray scale.

        Args:
            key: JAX PRNG key.
            shape: Leading dimensions of normals, i.e. normals.shape[:-1].
            element_idx: Passed through to :meth:`_gather`.

        Returns:
            angles: (*shape, 2) random angle pairs.
            scale: Broadcastable to (*shape, 1), in radians.
        """
        ...

    def perturb_normals(self, normals, key, element_idx=None):
        """Perturb surface normals.

        Works for any leading shape: ``(n_rays, 3)`` with per-ray
        *element_idx*, or ``(N, S, 3)`` with *element_idx=None* when
        the element dimension is already present.

        Args:
            normals: (..., 3).
            key: JAX PRNG key.
            element_idx: Per-ray element index, or None for batch mode.

        Returns:
            Perturbed normals (..., 3).
        """
        angles, scale = self._sample_perturbation(key, normals.shape[:-1], element_idx)
        return _apply_perturbation(normals, angles, scale)


# Gaussian BSDF

class GaussianBSDF(BSDF):
    """Single-Gaussian surface roughness model.

    Perturbs surface normals by Gaussian-distributed random angles.
    This is the standard model for surface microroughness.

    Attributes:
        scale: Per-element roughness sigma in arcseconds (N,).
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
    Gaussian with probability ``(1 - mix_weight)`` or from the wide
    Gaussian with probability ``mix_weight``.

    Attributes:
        scale_narrow: Per-element narrow-component sigma in arcseconds (N,).
        scale_wide: Per-element wide-component sigma in arcseconds (N,).
        mix_weight: Per-element probability of the wide component (N,),
                    values in [0, 1].
    """

    scale_narrow: Array  # (N,) in arcseconds
    scale_wide: Array    # (N,) in arcseconds
    mix_weight: Array    # (N,)

    def _sample_perturbation(self, key, shape, element_idx):
        k_angles, k_select = jr.split(key)
        angles = jr.normal(k_angles, (*shape, 2))

        wide = jr.uniform(k_select, shape) < self._gather(self.mix_weight, element_idx)
        sn = self._gather(self.scale_narrow, element_idx)
        sw = self._gather(self.scale_wide, element_idx)
        scale = jnp.where(wide, sw, sn)[..., None] * _ARCSEC_TO_RAD

        return angles, scale
