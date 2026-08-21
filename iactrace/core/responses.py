from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ._tolerances import dir_tol
from .ray_bundle import DEFAULT_WAVELENGTH


def fresnel_unpolarized(cos_theta_i, n1, n2):
    """Unpolarized Fresnel reflection and transmission coefficients.

    The standard formula for an ideal bare dielectric interface, used
    as the implicit default by :class:`RefractInteraction` and
    :class:`SlabInteraction` when no explicit :class:`ResponseCurve` is
    supplied. Average of s- and p-polarized intensities.

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

    def _reflectance(num, den):
        grazing = jnp.abs(den) < dir_tol(den)
        return jnp.where(grazing, 1.0, (num / jnp.where(grazing, 1.0, den)) ** 2)

    rs = _reflectance(n1 * cos_theta_i - n2 * cos_theta_t, n1 * cos_theta_i + n2 * cos_theta_t)
    rp = _reflectance(n2 * cos_theta_i - n1 * cos_theta_t, n2 * cos_theta_i + n1 * cos_theta_t)

    R = 0.5 * (rs + rp)
    return R, 1.0 - R


class ResponseCurve(eqx.Module):
    """Abstract base for an optical element's ``R(theta, lambda)`` response.

    A response curve maps each ray's incidence-angle cosine
    and wavelength to a coefficient in ``[0, 1]``.

    All subclasses return an array broadcastable to ``cos_theta_i.shape``.
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

    Attributes:
        values: Per-element coefficient in ``[0, 1]``, shape ``(N,)``.
    """

    values: Array  # (N,)

    def __call__(self, cos_theta_i, element_idx, wavelength=None):
        return self.values[element_idx]


class TabulatedResponse(ResponseCurve):
    """Bilinear interpolation over a shared ``(angle, wavelength)`` grid.

    Each ray's coefficient is read from a per-element
    ``cos(angle) x wavelength`` table, linearly interpolated in both axes
    and clamped at the grid edges (matching :func:`jax.numpy.interp`).

    Attributes:
        cos_table: ``cos(angle)`` axis, sorted ascending, shape ``(Kc,)``.
            ``cos_theta_i = 1`` -> normal incidence, ``0`` -> grazing.
        wl_table: wavelength axis, sorted ascending, shape ``(Kw,)``. Same
            units as :attr:`~iactrace.core.ray_bundle.RayBundle.wavelength`.
            Length 1 for a wavelength-independent curve.
        values: Per-element coefficient grid, shape ``(N, Kc, Kw)``.
    """

    cos_table: Array  # (Kc,)
    wl_table: Array  # (Kw,)
    values: Array  # (N, Kc, Kw)

    def __call__(self, cos_theta_i, element_idx, wavelength=None):
        rows = self.values[element_idx]  # (n_rays, Kc, Kw)

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
        """Build a :class:`TabulatedResponse` from human-readable angles.

        Args:
            angles_deg: Sample angles in degrees, shape ``(Kc,)`` (need not
                be sorted -- reordered into cos-ascending form internally).
            values: Coefficient values. Without ``wavelengths`` this is an
                angle curve: ``(Kc,)`` broadcast to all elements, or
                ``(N, Kc)`` per element. With ``wavelengths`` it is an
                ``(angle, wavelength)`` grid: ``(Kc, Kw)`` broadcast, or
                ``(N, Kc, Kw)`` per element.
            n_elements: Number of elements ``N`` in the enclosing group.
            wavelengths: Optional sample wavelengths ``(Kw,)`` (sorted
                internally). Omit for a wavelength-independent curve.

        Returns:
            A ready-to-use curve with cos- and wavelength-ascending
            lookup tables precomputed.
        """
        angles_deg = jnp.asarray(angles_deg)
        cos_table = jnp.cos(jnp.deg2rad(angles_deg))
        cos_order = jnp.argsort(cos_table)
        cos_table = cos_table[cos_order]

        v = jnp.asarray(values)
        if wavelengths is None:
            # Angle-only curve -> degenerate single-wavelength grid (Kw = 1).
            wl_table = jnp.asarray([DEFAULT_WAVELENGTH])
            if v.ndim == 1:
                v = jnp.broadcast_to(v, (n_elements, v.shape[0]))
            elif v.ndim == 2:
                if v.shape[0] != n_elements:
                    raise ValueError(
                        f"values first axis ({v.shape[0]}) must match n_elements ({n_elements})"
                    )
            else:
                raise ValueError(f"values must be 1D (Kc,) or 2D (N, Kc), got shape {v.shape}")
            if v.shape[1] != cos_table.shape[0]:
                raise ValueError(
                    f"values angle axis ({v.shape[1]}) must match angles_deg "
                    f"length ({cos_table.shape[0]})"
                )
            v = v[:, cos_order][:, :, None]  # (N, Kc, 1)
            return cls(cos_table=cos_table, wl_table=wl_table, values=v)

        wl_table = jnp.asarray(wavelengths)
        wl_order = jnp.argsort(wl_table)
        wl_table = wl_table[wl_order]
        if v.ndim == 2:
            v = jnp.broadcast_to(v, (n_elements, v.shape[0], v.shape[1]))
        elif v.ndim == 3:
            if v.shape[0] != n_elements:
                raise ValueError(
                    f"values first axis ({v.shape[0]}) must match n_elements ({n_elements})"
                )
        else:
            raise ValueError(f"values must be 2D (Kc, Kw) or 3D (N, Kc, Kw), got shape {v.shape}")
        if v.shape[1] != cos_table.shape[0] or v.shape[2] != wl_table.shape[0]:
            raise ValueError(
                f"values grid ({v.shape[1]}, {v.shape[2]}) must match "
                f"(angles_deg, wavelengths) = ({cos_table.shape[0]}, {wl_table.shape[0]})"
            )
        v = v[:, cos_order, :][:, :, wl_order]
        return cls(cos_table=cos_table, wl_table=wl_table, values=v)

    @classmethod
    def from_wavelengths(cls, wavelengths, values, n_elements: int) -> TabulatedResponse:
        """Build an **angle-flat** ``R(lambda)`` curve from wavelength samples.

        The convenience wrapper for the wavelength-only case.

        Args:
            wavelengths: Sample wavelengths ``(Kw,)`` (sorted internally). Same
                units as :attr:`~iactrace.core.ray_bundle.RayBundle.wavelength`.
            values: Coefficients in ``[0, 1]``: ``(Kw,)`` broadcast to all
                elements, or ``(N, Kw)`` per element.
            n_elements: Number of elements ``N`` in the enclosing group.

        Returns:
            An angle-independent :class:`TabulatedResponse` whose value depends
            only on wavelength.
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
