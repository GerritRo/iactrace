from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .transforms import euler_to_matrix

DEFAULT_WAVELENGTH = 400.0


class RayBundle(eqx.Module):
    """Bundle of rays through the optical system.

    The frame of origins / directions is implicit and depends on where
    the bundle came from: Telescope.render and Telescope.trace return
    rays in the camera-local frame, ready for Camera.collect /
    Camera.image.

    By the time the bundle reaches Camera.collect the entries of values
    are photoelectrons, not raw photons.

    Attributes
    ----------
    origins : array, shape (n_rays, 3)
        Ray positions. Meaningful only where alive.
    directions : array, shape (n_rays, 3)
        Ray directions. Meaningful only where alive.
    values : array, shape (n_rays,)
        Throughput-weighted intensities.
    path_length : array, shape (n_rays,)
        Accumulated optical path length, in metres.
    n : array, shape (n_rays,)
        Refractive index of the medium each ray is in, carried so downstream
        consumers can weight the final geometric leg.
    wavelength : array, shape (n_rays,)
        Per-ray wavelength.
    alive : array, shape (n_rays,)
        Per-ray liveness, all-True at construction.
    """

    origins: Array
    directions: Array
    values: Array
    path_length: Array
    n: Array
    wavelength: Array
    alive: Array

    def __init__(
        self,
        origins: Array,
        directions: Array,
        values: Array,
        path_length: Array,
        n: Array,
        alive: Array | None = None,
        wavelength: Array | None = None,
    ) -> None:
        self.origins = origins
        self.directions = directions
        self.values = values
        self.path_length = path_length
        self.n = n
        self.wavelength = (
            jnp.full(values.shape[0], DEFAULT_WAVELENGTH)
            if wavelength is None
            else jnp.asarray(wavelength)
        )
        self.alive = (
            jnp.ones(values.shape[0], dtype=bool)
            if alive is None
            else jnp.asarray(alive, dtype=bool)
        )

    def replace(self, **changes: Array) -> RayBundle:
        """Copy with the given fields replaced; unknown names raise TypeError."""
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        unknown = set(changes) - set(fields)
        if unknown:
            raise TypeError(f"RayBundle has no field(s) {sorted(unknown)}")
        return RayBundle(**{**fields, **changes})

    def to_frame(self, origin: Array, rotation: Array) -> RayBundle:
        """Transform rays to the frame given by origin + Euler rotation."""
        rot = euler_to_matrix(rotation)
        return self.replace(
            origins=(self.origins - origin) @ rot,
            directions=self.directions @ rot,
        )
