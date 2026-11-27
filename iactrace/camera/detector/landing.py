from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

from ...core.interpolation import bracket


def _safe_norm(v: Array) -> Array:
    """Vector length with a finite gradient at zero."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + 1e-30)


def landing_coordinates(
    origins: Array, directions: Array, normals: Array
) -> tuple[Array, Array, Array]:
    """Where and from which direction rays land on a photodetector face.

    The face is a surface of revolution about the pixel-local z axis (flat or
    domed), with outward normals pointing toward the incoming light.

    Parameters
    ----------
    origins : ndarray (N, 3)
        Landing points in the pixel-local frame.
    directions : ndarray (N, 3)
        Unit travel directions of the rays.
    normals : ndarray (N, 3)
        Outward unit surface normals at the landing points.

    Returns
    -------
    incidence
        Angle between the ray and the local normal in degrees (0 = normal
        incidence).
    radius
        Distance of the landing point from the axis.
    azimuth
        Direction of the ray around the local normal, in degrees in [0, 180]:
        0 = its tangential part heads radially outward (away from the axis),
        180 = toward the axis.
    """
    cos_i = -jnp.sum(directions * normals, axis=-1)
    d_tan = directions + cos_i[..., None] * normals
    incidence = jnp.arctan2(_safe_norm(d_tan), cos_i)
    radial = origins * jnp.array([1.0, 1.0, 0.0])
    radius = _safe_norm(radial)
    # Outward radial direction projected onto the tangent plane (unnormalised).
    outward = radial - jnp.sum(radial * normals, axis=-1, keepdims=True) * normals
    azimuth = jnp.arctan2(_safe_norm(jnp.cross(outward, d_tan)), jnp.sum(outward * d_tan, axis=-1))
    return jnp.degrees(incidence), radius, jnp.degrees(azimuth)


def _fractional_index(x: Array, grid: Array) -> Array:
    """Position of x in grid as a fractional index (clamped at the ends)."""
    lo, _, weight = bracket(grid, x)
    return lo + weight


class AngularResponse(eqx.Module):
    """Relative photocathode response against photon incidence angle.

    A table over incidence angle and, optionally, landing radius, direction
    azimuth (see landing_coordinates) and wavelength. Lookup is multilinear,
    clamped at the grid edges.

    Values are relative and not bounded by 1; the usual normalisation is 1
    for the beam the datasheet QE is measured with.

    Parameters
    ----------
    angles_deg
        Incidence angles to the local face normal, in degrees.
    values
        Response at the grid nodes, one dimension per given axis in the
        order (angle, radius, azimuth, wavelength).
    radius, azimuth_deg, wavelengths_nm
        Optional axes: landing distance from the tube axis, direction
        around the local normal in degrees (0 = heading away from the axis,
        180 = toward it), and wavelength.
    """

    angles_deg: Array
    radius: Array
    azimuth_deg: Array
    wavelengths_nm: Array
    values: Array

    def __init__(
        self, angles_deg, values, *, radius=None, azimuth_deg=None, wavelengths_nm=None
    ) -> None:
        names = ("angles_deg", "radius", "azimuth_deg", "wavelengths_nm")
        given = (angles_deg, radius, azimuth_deg, wavelengths_nm)
        axes = [
            jnp.zeros(1) if a is None else jnp.asarray(a, dtype=float).reshape(-1) for a in given
        ]
        values = jnp.asarray(values, dtype=float)
        shape = tuple(a.shape[0] for a, g in zip(axes, given, strict=True) if g is not None)
        if values.shape != shape:
            raise ValueError(f"values must have shape {shape}, got {values.shape}")
        for name, a in zip(names, axes, strict=True):
            if a.shape[0] > 1 and not bool(jnp.all(jnp.diff(a) > 0)):
                raise ValueError(f"{name} must be strictly increasing")
        self.angles_deg, self.radius, self.azimuth_deg, self.wavelengths_nm = axes
        self.values = values.reshape(tuple(a.shape[0] for a in axes))

    def __call__(self, incidence: Array, radius: Array, azimuth: Array, wavelength: Array) -> Array:
        """Response for rays with the given landing coordinates and wavelengths."""
        coords = [
            _fractional_index(incidence, self.angles_deg),
            _fractional_index(radius, self.radius),
            _fractional_index(azimuth, self.azimuth_deg),
            _fractional_index(wavelength, self.wavelengths_nm),
        ]
        return map_coordinates(self.values, coords, order=1, mode="nearest")

    def at(self, origins: Array, directions: Array, normals: Array, wavelength: Array) -> Array:
        """Response for rays landing at origins with directions on a surface with normals."""
        return self(*landing_coordinates(origins, directions, normals), wavelength)
