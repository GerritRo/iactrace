from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def fresnel_unpolarized(cos_theta_i, n1, n2):
    """Unpolarized Fresnel reflection and transmission coefficients.

    The standard formula for an ideal bare dielectric interface, used
    as the implicit default by :class:`RefractInteraction` and
    :class:`SlabInteraction` when no explicit :class:`Coating` is
    supplied. Average of s- and p-polarized intensities.

    The transmitted angle is derived internally from Snell's law, so
    only the incidence cosine and the two indices are needed. Total
    internal reflection (``sin^2(theta_t) > 1``) collapses ``cos_theta_t`` to
    zero, which the formula correctly turns into ``R = 1, T = 0``.

    Args:
        cos_theta_i: Cosine of the incidence angle.
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitted medium.

    Returns:
        R: Reflectance in ``[0, 1]``.
        T: Transmittance ``T = 1 - R``.
    """
    eta = n1 / n2
    sin2_t = eta**2 * (1.0 - cos_theta_i**2)
    cos_theta_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))

    rs_num = n1 * cos_theta_i - n2 * cos_theta_t
    rs_den = n1 * cos_theta_i + n2 * cos_theta_t
    rs_den = jnp.where(jnp.abs(rs_den) < 1e-10, 1e-10, rs_den)
    rs = (rs_num / rs_den)**2

    rp_num = n2 * cos_theta_i - n1 * cos_theta_t
    rp_den = n2 * cos_theta_i + n1 * cos_theta_t
    rp_den = jnp.where(jnp.abs(rp_den) < 1e-10, 1e-10, rp_den)
    rp = (rp_num / rp_den)**2

    R = 0.5 * (rs + rp)
    return R, 1.0 - R


class Coating(eqx.Module):
    """Abstract base for angle-dependent optical coatings.

    A coating maps the incidence-angle cosine of each ray to a
    coefficient in ``[0, 1]`` (reflectance or transmittance, depending
    on the surface type). All subclasses must return an array
    broadcastable to ``cos_theta_i.shape``.
    """

    @abstractmethod
    def __call__(
        self,
        cos_theta_i: Array,
        element_idx: Array,
    ) -> Array: ...


class ConstantCoating(Coating):
    """Angle-independent per-element coating.

    Attributes:
        values: Per-element coefficient in ``[0, 1]``, shape ``(N,)``.
    """

    values: Array  # (N,)

    def __call__(self, cos_theta_i, element_idx):
        return self.values[element_idx]


class TabulatedCoating(Coating):
    """Linear interpolation over a shared angle grid.

    Attributes:
        cos_table: ``cos(angle)`` lookup axis, sorted ascending, shape
            ``(K,)``. ``cos_theta_i = 1`` -> normal incidence,
            ``cos_theta_i = 0`` -> grazing.
        values: Per-element coefficient values aligned with
            ``cos_table``, shape ``(N, K)``.
    """

    cos_table: Array  # (K,)
    values: Array     # (N, K)

    def __call__(self, cos_theta_i, element_idx):
        rows = self.values[element_idx]  # (n_rays, K)
        return jax.vmap(
            lambda c, r: jnp.interp(c, self.cos_table, r)
        )(cos_theta_i, rows)

    @classmethod
    def from_degrees(
        cls,
        angles_deg,
        values,
        n_elements: int,
    ) -> TabulatedCoating:
        """Build a :class:`TabulatedCoating` from human-readable angles.

        Args:
            angles_deg: Sample angles in degrees, shape ``(K,)``. Don't 
                need to be sorted since they are reordered into cos-ascending
                form internally.
            values: Coefficient values. ``(K,)`` is broadcast to all
                ``n_elements`` elements; ``(N, K)`` is used as-is and
                must match ``n_elements`` along the first axis.
            n_elements: Number of elements ``N`` in the enclosing group.

        Returns:
            A ready-to-use coating with the cos-ascending lookup table
            precomputed.
        """
        angles_deg = jnp.asarray(angles_deg)
        cos_table = jnp.cos(jnp.deg2rad(angles_deg))
        order = jnp.argsort(cos_table)
        cos_table = cos_table[order]

        v = jnp.asarray(values)
        if v.ndim == 1:
            v = jnp.broadcast_to(v, (n_elements, v.shape[0]))
        elif v.ndim == 2:
            if v.shape[0] != n_elements:
                raise ValueError(
                    f"values first axis ({v.shape[0]}) must match "
                    f"n_elements ({n_elements})"
                )
        else:
            raise ValueError(
                f"values must be 1D (K,) or 2D (N, K), got shape {v.shape}"
            )
        v = v[:, order]
        return cls(cos_table=cos_table, values=v)