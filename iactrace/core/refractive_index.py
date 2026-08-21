from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .ray_bundle import DEFAULT_WAVELENGTH


class RefractiveIndex(eqx.Module):
    """A refracting element's index as a function of wavelength, ``n(lambda)``.

    Subclasses provide:

    * :meth:`n_at` -- per-ray index ``(n_rays,)`` at the given wavelengths,
      selecting per-element parameters with ``element_idx``.
    * :meth:`reference` -- a per-element nominal index ``(N,)`` at a single
      *design* wavelength, used by the focal-length <-> curvature maths
      (:func:`~iactrace.telescope.operations.set_focal_lengths`).
    * :attr:`n_elements` -- the number of elements ``N`` the model sizes.
    """

    @abstractmethod
    def n_at(self, element_idx: Array, wavelength: Array) -> Array: ...

    @abstractmethod
    def reference(self, wavelength: float | Array = DEFAULT_WAVELENGTH) -> Array: ...

    @property
    @abstractmethod
    def n_elements(self) -> int: ...


class ConstantIndex(RefractiveIndex):
    """Wavelength-independent per-element index (the monochromatic case).

    Attributes:
        values: Per-element refractive index, shape ``(N,)``.
    """

    values: Array  # (N,)

    def n_at(self, element_idx, wavelength):
        return self.values[element_idx]

    def reference(self, wavelength=DEFAULT_WAVELENGTH):
        return self.values

    @property
    def n_elements(self):
        return self.values.shape[0]


class TabulatedIndex(RefractiveIndex):
    """Per-element refractive index, linearly interpolated in wavelength.

    Attributes:
        wavelengths: Lookup axis, sorted ascending, shape ``(K,)``. Same
            units as :attr:`~iactrace.core.ray_bundle.RayBundle.wavelength`.
        n_values: Per-element index samples aligned with ``wavelengths``,
            shape ``(N, K)``.
    """

    wavelengths: Array  # (K,)
    n_values: Array  # (N, K)

    def n_at(self, element_idx, wavelength):
        rows = self.n_values[element_idx]  # (n_rays, K)
        return jax.vmap(lambda w, r: jnp.interp(w, self.wavelengths, r))(wavelength, rows)

    def reference(self, wavelength=DEFAULT_WAVELENGTH):
        return jax.vmap(lambda r: jnp.interp(wavelength, self.wavelengths, r))(self.n_values)

    @property
    def n_elements(self):
        return self.n_values.shape[0]

    @classmethod
    def from_table(cls, wavelengths, n_values, n_elements: int) -> TabulatedIndex:
        """Build from measured ``n(lambda)`` samples.

        Args:
            wavelengths: Sample wavelengths, shape ``(K,)`` (sorted
                internally).
            n_values: Index samples. ``(K,)`` is broadcast to all
                ``n_elements`` elements; ``(N, K)`` is used as-is.
            n_elements: Number of elements ``N`` in the group.
        """
        wl = jnp.asarray(wavelengths)
        order = jnp.argsort(wl)
        wl = wl[order]

        v = jnp.asarray(n_values)
        if v.ndim == 1:
            v = jnp.broadcast_to(v, (n_elements, v.shape[0]))
        elif v.ndim == 2:
            if v.shape[0] != n_elements:
                raise ValueError(
                    f"n_values first axis ({v.shape[0]}) must match n_elements ({n_elements})"
                )
        else:
            raise ValueError(f"n_values must be 1D (K,) or 2D (N, K), got shape {v.shape}")
        if v.shape[-1] != wl.shape[0]:
            raise ValueError(
                f"n_values wavelength axis ({v.shape[-1]}) must match "
                f"wavelengths length ({wl.shape[0]})"
            )
        v = v[:, order]
        return cls(wavelengths=wl, n_values=v)


class SellmeierIndex(RefractiveIndex):
    """Sellmeier-equation refractive index per element.

    ``n(lambda)^2 = 1 + sum_j b_j * lambda^2 / (lambda^2 - c_j)``.

    The ``b`` coefficients are dimensionless; the ``c`` coefficients carry
    units of wavelength squared and **must match the unit of
    :attr:`~iactrace.core.ray_bundle.RayBundle.wavelength`**.

    Attributes:
        b: Per-element Sellmeier B coefficients, shape ``(N, M)``.
        c: Per-element Sellmeier C coefficients, shape ``(N, M)``.
    """

    b: Array  # (N, M)
    c: Array  # (N, M)

    def _n(self, b, c, wavelength):
        lam2 = (wavelength**2)[..., None]
        return jnp.sqrt(1.0 + jnp.sum(b * lam2 / (lam2 - c), axis=-1))

    def n_at(self, element_idx, wavelength):
        return self._n(self.b[element_idx], self.c[element_idx], wavelength)

    def reference(self, wavelength=DEFAULT_WAVELENGTH):
        wl = jnp.full(self.b.shape[0], wavelength)
        return self._n(self.b, self.c, wl)

    @property
    def n_elements(self):
        return self.b.shape[0]


def as_refractive_index(index, n_elements: int) -> RefractiveIndex:
    """Coerce an index argument to a :class:`RefractiveIndex` sized ``n_elements``.

    Args:
        index: A :class:`RefractiveIndex` model, a scalar, or a per-element
            ``(N,)`` array.
        n_elements: Number of elements ``N`` the model must size.

    Raises:
        ValueError: if ``index`` is ``None``, or a model sized for a
            different number of elements.
    """
    if index is None:
        raise ValueError(
            "an `index` is required: pass a number for a non-dispersive "
            "element, or a RefractiveIndex model for a dispersive one"
        )
    if isinstance(index, RefractiveIndex):
        if index.n_elements != n_elements:
            raise ValueError(f"index model has {index.n_elements} elements, expected {n_elements}")
        return index
    return ConstantIndex(jnp.broadcast_to(jnp.asarray(index, dtype=float), (n_elements,)))
