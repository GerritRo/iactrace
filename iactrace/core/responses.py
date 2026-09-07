from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .ray_bundle import DEFAULT_WAVELENGTH
from .tolerances import dir_tol


def fresnel_unpolarized(cos_theta_i, n1, n2):
    """Unpolarized Fresnel coefficients (R, T) for a bare dielectric interface.

    The average of s- and p-polarized intensities, with T = 1 - R. Used as
    the implicit default by RefractInteraction and
    SlabInteraction when no explicit ResponseCurve is given.
    """
    eta = n1 / n2
    sin2_t = eta**2 * (1.0 - cos_theta_i**2)
    cos_theta_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))

    def _reflectance(num, den):
        grazing = jnp.abs(den) < dir_tol(den)
        return jnp.where(grazing, 1.0, (num / jnp.where(grazing, 1.0, den)) ** 2)

    rs = _reflectance(n1 * cos_theta_i - n2 * cos_theta_t, n1 * cos_theta_i + n2 * cos_theta_t)
    rp = _reflectance(n2 * cos_theta_i - n1 * cos_theta_t, n2 * cos_theta_i + n1 * cos_theta_t)

    R = 0.5 * (rs + rp)
    return R, 1.0 - R


def _broadcast_to_elements(values, n_elements, shared_ndim, label):
    """Bring values to a leading per-element axis of length n_elements."""
    v = jnp.asarray(values)
    if v.ndim == shared_ndim:
        return jnp.broadcast_to(v, (n_elements, *v.shape))
    if v.ndim == shared_ndim + 1:
        if v.shape[0] != n_elements:
            raise ValueError(
                f"values first axis ({v.shape[0]}) must match n_elements ({n_elements})"
            )
        return v
    raise ValueError(f"values must be {label}, got shape {v.shape}")


class ResponseCurve(eqx.Module):
    """Abstract base for an optical elements R(theta, lambda) response.

    A response curve maps each rays incidence-angle cosine
    and wavelength to a coefficient in [0, 1].

    All subclasses return an array broadcastable to cos_theta_i.shape.
    """

    @abstractmethod
    def __call__(
        self,
        cos_theta_i: Array,
        element_idx: Array,
        wavelength: Array | None = None,
    ) -> Array: ...


class ConstantResponse(ResponseCurve):
    """Angle- and wavelength-independent per-element response.

    Attributes
    ----------
    values : array, shape (N,)
        Per-element coefficient in [0, 1].
    """

    values: Array  # (N,)

    def __call__(self, cos_theta_i, element_idx, wavelength=None):
        return self.values[element_idx]


class TabulatedResponse(ResponseCurve):
    """Bilinear interpolation over a shared (angle, wavelength) grid.

    Each ray's coefficient is read from a per-element
    cos(angle) x wavelength table, linearly interpolated in both axes
    and clamped at the grid edges (matching jax.numpy.interp).

    Attributes
    ----------
    cos_table : array, shape (Kc,)
        cos(angle) axis, sorted ascending.
        cos_theta_i = 1 -> normal incidence, 0 -> grazing.
    wl_table : array, shape (Kw,)
        Wavelength axis, sorted ascending. Same units as wavelength. Length 1 for a
        wavelength-independent curve.
    values : array, shape (N, Kc, Kw)
        Per-element coefficient grid.
    """

    cos_table: Array  # (Kc,)
    wl_table: Array   # (Kw,)
    values: Array     # (N, Kc, Kw)

    def __call__(self, cos_theta_i, element_idx, wavelength=None):
        rows = self.values[element_idx]

        # Wavelength-independent curve: skip and interpolate in cos only
        if self.wl_table.shape[0] == 1:
            return jax.vmap(lambda c, r: jnp.interp(c, self.cos_table, r[:, 0]))(cos_theta_i, rows)

        if wavelength is None:
            wavelength = jnp.full_like(cos_theta_i, self.wl_table[0])

        def per_ray(cos, wl, grid):
            col = jax.vmap(lambda r: jnp.interp(wl, self.wl_table, r))(grid)  # (Kc,)
            return jnp.interp(cos, self.cos_table, col)

        return jax.vmap(per_ray)(cos_theta_i, wavelength, rows)

    @classmethod
    def from_degrees(
        cls,
        angles_deg,
        values,
        n_elements: int,
        *,
        wavelengths=None,
    ) -> TabulatedResponse:
        """Build a TabulatedResponse from angles given in degrees.

        Both axes are sorted internally, so angles_deg (Kc,) and
        wavelengths (Kw,) may arrive in any order.

        Parameters
        ----------
        values : array, shape (Kc,)
            Without wavelengths, an angle curve --.
            broadcast to all elements, or (N, Kc) per element. With
            wavelengths, an (angle, wavelength) grid -- (Kc, Kw)
            broadcast, or (N, Kc, Kw) per element.
        wavelengths
            Omit for a wavelength-independent curve.
        """
        angles_deg = jnp.asarray(angles_deg)
        cos_table = jnp.cos(jnp.deg2rad(angles_deg))
        cos_order = jnp.argsort(cos_table)
        cos_table = cos_table[cos_order]

        if wavelengths is None:
            # Angle-only curve -> degenerate single-wavelength grid (Kw = 1).
            wl_table = jnp.asarray([DEFAULT_WAVELENGTH])
            wl_order = jnp.zeros(1, dtype=int)
            v = _broadcast_to_elements(values, n_elements, 1, "1D (Kc,) or 2D (N, Kc)")
            v = v[:, :, None]
        else:
            wl_table = jnp.asarray(wavelengths)
            wl_order = jnp.argsort(wl_table)
            wl_table = wl_table[wl_order]
            v = _broadcast_to_elements(values, n_elements, 2, "2D (Kc, Kw) or 3D (N, Kc, Kw)")
        if v.shape[1] != cos_table.shape[0]:
            raise ValueError(
                f"values angle axis ({v.shape[1]}) must match angles_deg "
                f"length ({cos_table.shape[0]})"
            )
        if v.shape[2] != wl_table.shape[0]:
            raise ValueError(
                f"values wavelength axis ({v.shape[2]}) must match wavelengths "
                f"length ({wl_table.shape[0]})"
            )
        v = v[:, cos_order, :][:, :, wl_order]
        return cls(cos_table=cos_table, wl_table=wl_table, values=v)

    @classmethod
    def from_wavelengths(cls, wavelengths, values, n_elements: int) -> TabulatedResponse:
        """Build an angle-flat R(lambda) curve from wavelength samples.

        The convenience wrapper for the wavelength-only case.
        """
        v = jnp.asarray(values)
        if v.ndim == 1:
            v = v[None, :]
        elif v.ndim == 2:
            v = v[:, None, :]
        else:
            raise ValueError(f"values must be 1D (Kw,) or 2D (N, Kw), got shape {v.shape}")
        return cls.from_degrees(
            angles_deg=[0.0], values=v, n_elements=n_elements, wavelengths=wavelengths
        )
