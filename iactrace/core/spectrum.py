from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class Spectrum(eqx.Module):
    """A source's distribution over wavelength.

    Renders consume a spectrum through sample, which gives each ray one
    wavelength drawn from the distribution -- so a broadband render costs
    exactly as many rays as a monochromatic one, and the band is integrated by
    the Monte Carlo ensemble. The draw is reparameterised (inverse-CDF of a
    fixed uniform stream), so gradients flow to the spectrum's own parameters.

    bins is the quadrature counterpart, for callers who would rather
    sweep deterministically over (wavelengths, weights).
    """

    @abstractmethod
    def sample(self, key: Array, shape: tuple[int, ...]) -> Array:
        """Draw wavelengths of the given shape from the distribution."""

    @abstractmethod
    def bins(self) -> tuple[Array, Array]:
        """Return (wavelengths, weights) for a quadrature sweep.

        weights are normalised (sum to 1) so a weighted sum of
        per-wavelength renders estimates the flux-averaged broadband result.
        """


class ConstantSpectrum(Spectrum):
    """Monochromatic source: every ray at a single wavelength (degenerate case).

    Attributes
    ----------
    wavelength
        The single scalar wavelength.
    """

    wavelength: Array  # scalar

    def sample(self, key, shape):
        return jnp.full(shape, self.wavelength)

    def bins(self):
        return jnp.reshape(self.wavelength, (1,)), jnp.ones(1)


class TabulatedSpectrum(Spectrum):
    """Piecewise-linear photon density sampled at a set of wavelengths.

    density is the relative spectral photon density at wavelengths
    (any non-negative scale; it is normalised internally). sample draws
    from that density by exact inverse-CDF; bins returns the same
    density as trapezoidal quadrature weights.

    Attributes
    ----------
    wavelengths : ndarray (K,)
        Sample wavelengths, strictly ascending. At least two are required.
    density : ndarray (K,)
        Relative photon density aligned with wavelengths, non-negative and not
        zero everywhere.

    Raises
    ------
    ValueError
        if the two arrays disagree in shape, if fewer than two wavelengths are
        given, or if the wavelengths are not strictly ascending or the density
        carries no mass.
    """

    wavelengths: Array
    density: Array

    def __check_init__(self):
        if self.wavelengths.ndim != 1 or self.density.ndim != 1:
            raise ValueError(
                "`wavelengths` and `density` must both be 1-D, got shapes "
                f"{self.wavelengths.shape} and {self.density.shape}"
            )
        if self.wavelengths.shape != self.density.shape:
            raise ValueError(
                f"`density` ({self.density.shape[0]}) must give one value per wavelength "
                f"({self.wavelengths.shape[0]})"
            )
        if self.wavelengths.shape[0] < 2:
            raise ValueError("`wavelengths` must give at least two values.")
        # Value checks only where the arrays are concrete
        if isinstance(self.wavelengths, jax.core.Tracer) or isinstance(
            self.density, jax.core.Tracer
        ):
            return
        if not bool(jnp.all(jnp.diff(self.wavelengths) > 0.0)):
            raise ValueError("`wavelengths` must be strictly ascending with no repeats")
        if not bool(jnp.all(self.density >= 0.0)):
            raise ValueError("`density` must be non-negative")
        if not bool(jnp.sum(self.density) > 0.0):
            raise ValueError("`density` is zero everywhere.")

    def _cdf(self):
        dwl = jnp.diff(self.wavelengths)
        seg = 0.5 * (self.density[1:] + self.density[:-1]) * dwl
        cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
        return cdf / cdf[-1]

    def sample(self, key, shape):
        """Draw wavelengths from the piecewise-linear density.

        Reparameterised so gradients flow to the spectrum's own parameters.
        """
        u = jax.random.uniform(key, shape)
        cdf = self._cdf()
        i = jnp.clip(jnp.searchsorted(cdf, u, side="right") - 1, 0, self.wavelengths.shape[0] - 2)
        p0, p1 = self.density[i], self.density[i + 1]
        lo, hi = cdf[i], cdf[i + 1]
        wide = hi > lo
        frac = jnp.where(wide, (u - lo) / jnp.where(wide, hi - lo, 1.0), 0.0)

        disc = p0 * p0 + (p1 - p0) * frac * (p0 + p1)
        pos = disc > 0.0
        root = jnp.where(pos, jnp.sqrt(jnp.where(pos, disc, 1.0)), 0.0)
        den = p0 + root
        live = den > 0.0
        t = jnp.where(live, frac * (p0 + p1) / jnp.where(live, den, 1.0), 0.0)
        return self.wavelengths[i] + t * jnp.diff(self.wavelengths)[i]

    def bins(self):
        dwl = jnp.diff(self.wavelengths)
        w = jnp.zeros_like(self.density)
        w = w.at[:-1].add(0.5 * self.density[:-1] * dwl)  # trapezoidal node weights
        w = w.at[1:].add(0.5 * self.density[1:] * dwl)
        return self.wavelengths, w / w.sum()

    @classmethod
    def from_density(cls, wavelengths, density) -> TabulatedSpectrum:
        """Build from (wavelengths, density) samples (sorted internally)."""
        wl = jnp.asarray(wavelengths)
        d = jnp.asarray(density)
        order = jnp.argsort(wl)
        return cls(wavelengths=wl[order], density=d[order])


def as_spectrum(wavelength) -> Spectrum:
    """Coerce a wavelength argument to a Spectrum.

    A Spectrum is returned as-is; anything else (a scalar) becomes a
    ConstantSpectrum.
    """
    if isinstance(wavelength, Spectrum):
        return wavelength
    return ConstantSpectrum(jnp.asarray(wavelength))
