from __future__ import annotations

import dataclasses
import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.spectrum import TabulatedSpectrum
from ..core.transforms import euler_to_matrix

__all__ = [
    "EffectiveApertureTable",
    "FieldFrame",
    "effective_aperture",
    "effective_area",
    "pixel_response",
]


# Field frame


class FieldFrame(eqx.Module):
    """The telescope's field-angle frame -- optical axis plus two sky axes.

    A field direction (lon, lat) is the propagation direction of a plane wave
    arriving from that offset in the usual spherical offset coordinates about the
    optical axis.

    Attributes
    ----------
    axis
        Unit propagation direction of an on-axis ray -- the direction
        light travels, pointing into the telescope (3,).
    e_lon : array, shape (3,)
        Unit sky axis of increasing lon.
    e_lat : array, shape (3,)
        Unit sky axis of increasing lat.
    """

    axis: Array  # (3,)
    e_lon: Array  # (3,)
    e_lat: Array  # (3,)

    @classmethod
    def from_telescope(cls, telescope, rotation_deg: float = 0.0) -> FieldFrame:
        """Derive the field frame from a telescope's optics and mount convention.

        Parameters
        ----------
        telescope
            The Telescope.
        rotation_deg
            Roll about the optical axis.
        """
        rotations = jax.vmap(euler_to_matrix)(jnp.asarray(telescope.stage(0).rotations))
        axis = np.asarray(rotations)[:, :, 2].mean(axis=0)
        axis = -axis / np.linalg.norm(axis)

        e_lon = _orthogonalise(np.array([0.0, 1.0, 0.0]), axis)  # +Y
        e_lat = np.cross(axis, e_lon)  # +X
        if rotation_deg:
            c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
            e_lon, e_lat = c * e_lon + s * e_lat, -s * e_lon + c * e_lat

        return cls(axis=jnp.asarray(axis), e_lon=jnp.asarray(e_lon), e_lat=jnp.asarray(e_lat))

    def directions(self, lon, lat) -> Array:
        """Unit propagation directions (..., 3) for field offsets in radians."""
        lon, lat = jnp.broadcast_arrays(jnp.asarray(lon), jnp.asarray(lat))
        return (
            (jnp.cos(lat) * jnp.cos(lon))[..., None] * self.axis
            + (jnp.cos(lat) * jnp.sin(lon))[..., None] * self.e_lon
            + jnp.sin(lat)[..., None] * self.e_lat
        )


def _orthogonalise(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of v perpendicular to axis, normalised."""
    residual = np.asarray(v, dtype=float) - np.dot(v, axis) * axis
    if np.linalg.norm(residual) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        residual = fallback - np.dot(fallback, axis) * axis
    return residual / np.linalg.norm(residual)


@eqx.filter_jit
def pixel_response(telescope, camera, directions, spectrum=None, *, sensor_idx=0):
    """Per-pixel effective area for field directions (n_directions, 3), in m^2."""
    rb = telescope.render(
        directions, jnp.ones(directions.shape[0]), source_type="parallel", wavelength=spectrum
    )
    matrix = camera.response_matrix(rb, sensor_idx)  # (n_dir, n_sensors, *pixel_shape)
    return matrix.reshape(directions.shape[0], -1)


def effective_area(
    telescope, camera, wavelengths, *, direction=None, sensor_idx=0, progress=False
) -> np.ndarray:
    """Total effective collecting area per wavelength in square metres."""
    if direction is None:
        direction = FieldFrame.from_telescope(telescope).axis
    direction = jnp.asarray(direction, dtype=float)[None, :]

    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    areas = np.empty(wavelengths.size)
    for k, wl in enumerate(wavelengths):
        response = pixel_response(
            telescope, camera, direction, jnp.asarray(wl), sensor_idx=sensor_idx
        )
        areas[k] = float(response.sum())
        if progress:
            _progress(f"effective_area: {k + 1}/{wavelengths.size} wavelengths")
    if progress:
        _progress("", end="\n")
    return areas


def _progress(message: str, end: str = "\r") -> None:
    print(f"{message:<70}", file=sys.stderr, end=end, flush=True)


def _centroid(origin, step, offset, values) -> np.ndarray:
    """Response-weighted field offset of each pixel (n_pixels, 2), radians."""
    total = values.sum(axis=(1, 2))
    nodes = np.arange(values.shape[-1], dtype=float)
    node = np.stack([values.sum(axis=2) @ nodes, values.sum(axis=1) @ nodes], axis=-1)
    node = node / np.where(total > 0, total, 1.0)[:, None]
    # A dark pixel has no first moment. Falls back to the window centre.
    node = np.where((total > 0)[:, None], node, (values.shape[-1] - 1) / 2.0)
    return origin + (offset + node) * step


@dataclasses.dataclass(frozen=True)
class EffectiveApertureTable:
    """A telescope's per-pixel effective aperture, tabulated over field angle.

    Node (i, j) sits at field offset origin + (i, j) * step, and pixel p
    covers nodes offset[p] to offset[p] + values.shape[-2:]. Field offsets
    are the spherical offset coordinates of FieldFrame, and
    lon = lat = 0 is a node by construction.

    Attributes
    ----------
    origin : array, shape (2,)
        Field offset of node (0, 0), [lon, lat] in radians.
    step : array, shape (2,)
        Node spacing along [lon, lat], in radians.
    offset : array, shape (n_pixels, 2)
        Node index of each pixel's window corner.
    values
        Effective area in m^2 (n_pixels, W, W); axis -2 runs along
        lon, axis -1 along lat. Zero on the boundary by construction.
    on_axis_area
        On-axis effective area over all pixels, band-averaged over
        spectral_area. Normalises values dimensionless.
    wavelengths : array, shape (K,)
        Wavelength grid of spectral_area, in nm.
    spectral_area : array, shape (K,)
        On-axis total effective area per wavelength.
    meta
        Provenance, including the table's own on-axis sum for comparison
        against on_axis_area (they differ only by Monte-Carlo noise).
    """

    origin: np.ndarray
    step: np.ndarray
    offset: np.ndarray
    values: np.ndarray
    on_axis_area: float
    wavelengths: np.ndarray
    spectral_area: np.ndarray
    meta: dict

    @property
    def n_pixels(self) -> int:
        """Number of pixels tabulated."""
        return int(self.values.shape[0])

    @property
    def window(self) -> int:
        """Side length of each pixel's response window, in lattice nodes."""
        return int(self.values.shape[-1])

    @property
    def centres(self) -> np.ndarray:
        """Field offset each pixel looks at (n_pixels, 2), radians."""
        return _centroid(self.origin, self.step, self.offset, self.values)

    @property
    def window_centres(self) -> np.ndarray:
        """Field offset of each pixel's window centre. Shape (n_pixels, 2), radians."""
        return self.origin + (self.offset + (self.window - 1) / 2.0) * self.step

    def save(self, filename, *, overwrite: bool = True, compress: bool = True):
        """Write to a .npz file; see iactrace.io.save_aperture_table."""
        from ..io.aperture_table import save_aperture_table

        return save_aperture_table(self, filename, overwrite=overwrite, compress=compress)

    @classmethod
    def load(cls, filename) -> EffectiveApertureTable:
        """Read a table back; see iactrace.io.load_aperture_table."""
        from ..io.aperture_table import load_aperture_table

        return load_aperture_table(filename)


def _bandpass(telescope, camera, frame, wavelengths, sensor_idx, progress):
    """Spectral bandpass of a telescope.

    Returns (wavelengths, spectral_area, spectrum, on_axis_area).
    """
    wavelengths = np.unique(np.asarray(wavelengths, dtype=float).reshape(-1))
    if wavelengths.size < 2:
        raise ValueError(
            "`wavelengths` must give at least two distinct values: a bandpass over a "
            "single wavelength is a spike, not a band, and cannot be integrated by a "
            "consumer. Two points spanning the range describe a flat response."
        )
    spectral_area = effective_area(
        telescope,
        camera,
        wavelengths,
        direction=frame.axis,
        sensor_idx=sensor_idx,
        progress=progress,
    )
    if not np.any(spectral_area > 0):
        raise ValueError(
            f"the telescope delivers no light to the camera anywhere in "
            f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm; check the optics, the sensor "
            f"placement, `sensor_idx`, and that `wavelengths` covers the optics' band"
        )

    spectrum = TabulatedSpectrum.from_density(
        jnp.asarray(wavelengths), jnp.asarray(np.clip(spectral_area, 0.0, None))
    )
    _, weights = spectrum.bins()
    return wavelengths, spectral_area, spectrum, float(np.sum(np.asarray(weights) * spectral_area))


def _extent(mask_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last True index along axis 0 of an (n, n_pixels) mask."""
    n = mask_2d.shape[0]
    idx = np.arange(n)[:, None]
    return np.where(mask_2d, idx, n - 1).min(axis=0), np.where(mask_2d, idx, 0).max(axis=0)


def _field(
    telescope, camera, frame, spectrum, sensor_idx, n_pix, half_angle, step, chunk, progress
):
    """Scan the whole lattice: the response at every field offset, for every pixel.

    Returns (cube, axis1d).
    """
    n = 2 * int(round(half_angle / step)) + 1
    axis1d = (np.arange(n) - n // 2) * step
    grid_lon, grid_lat = (a.ravel() for a in np.meshgrid(axis1d, axis1d, indexing="ij"))

    cube = np.zeros((n * n, n_pix), np.float32)
    total = grid_lon.size
    chunk = max(1, min(int(chunk), total))
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        idx = np.clip(np.arange(start, start + chunk), 0, total - 1)
        directions = frame.directions(jnp.asarray(grid_lon[idx]), jnp.asarray(grid_lat[idx]))
        response = pixel_response(telescope, camera, directions, spectrum, sensor_idx=sensor_idx)
        cube[start:stop] = np.asarray(response)[: stop - start]
        if progress:
            _progress(f"scan: {stop}/{total} directions")
    if progress:
        _progress("", end="\n")
    np.clip(cube, 0.0, None, out=cube)
    return cube.reshape(n, n, -1), axis1d


def _windows(cube, window, tol):
    """Bound every pixel's response, then slice one window per pixel out of cube.

    Returns (values, offset, window), offsets in node indices.
    """
    n, _, n_pix = cube.shape
    peak = cube.max(axis=(0, 1))
    if not np.all(peak > 0):
        raise ValueError(
            f"{int(np.sum(peak <= 0))} of {n_pix} pixels received no light; the scanned "
            f"field may be too small, or `step` too coarse to land on them"
        )
    lit_lon = cube.max(axis=1) > tol * peak
    lit_lat = cube.max(axis=0) > tol * peak
    if lit_lon[0].any() or lit_lon[-1].any() or lit_lat[0].any() or lit_lat[-1].any():
        raise ValueError("pixel response reaches the edge of the field; raise `half_angle`")

    lo_lon, hi_lon = _extent(lit_lon)
    lo_lat, hi_lat = _extent(lit_lat)
    lo = np.stack([lo_lon, lo_lat], axis=-1)
    hi = np.stack([hi_lon, hi_lat], axis=-1)
    if window is None:
        window = int((hi - lo).max()) + 3
    window = int(window) | 1  # odd, so Simpson quadrature over the window is exact
    if window > n:
        raise ValueError(f"a {window}x{window} window does not fit a {n}x{n} field")

    offset = np.clip((lo + hi - (window - 1)) // 2, 0, n - window)
    rows = offset[:, 0, None] + np.arange(window)
    cols = offset[:, 1, None] + np.arange(window)
    values = cube[rows[:, :, None], cols[:, None, :], np.arange(n_pix)[:, None, None]]

    border = np.concatenate(
        [values[:, 0, :], values[:, -1, :], values[:, 1:-1, 0], values[:, 1:-1, -1]], axis=1
    ).sum(axis=1)
    total = values.sum(axis=(1, 2))
    fraction = np.divide(border, total, out=np.zeros_like(border), where=total > 0)
    if float(fraction.max()) > tol:
        raise ValueError(
            f"pixel {int(fraction.argmax())} carries {float(fraction.max()):.2%} of its "
            f"response on the boundary of a {window}x{window} window, above tol={tol:.1e}; "
            f"pass a larger `window` (or a coarser `step`)"
        )
    values[:, 0, :] = values[:, -1, :] = 0.0
    values[:, :, 0] = values[:, :, -1] = 0.0
    return values, offset, window


def effective_aperture(
    telescope,
    camera,
    *,
    half_angle: float,
    step: float,
    wavelengths,
    sensor_idx: int = 0,
    window: int | None = None,
    frame: FieldFrame | None = None,
    chunk_size: int = 512,
    tol: float = 1e-4,
    progress: bool = False,
) -> EffectiveApertureTable:
    """Tabulate a telescope + camera's per-pixel effective aperture.

    Parameters
    ----------
    telescope
        Its n_samples per primary element sets the Monte-Carlo noise.
    camera
        The camera.
    half_angle
        Half-width of the field to scan, radians. Must clear the camera:
        roughly atan(camera_radius / focal_length) plus margin.
    step
        Lattice node spacing, radians. Resolves the response shape.
    wavelengths
        Grid (K,) for B(lambda), in nm; at least two distinct values.
    sensor_idx
        Which sensor group to tabulate.
    window
        Side length of each response window in nodes, forced odd. Defaults
        to the largest measured extent plus a zero border.
    frame
        The FieldFrame offsets are measured in. Defaults to
        FieldFrame.from_telescope.
    chunk_size
        Directions per render call; bounds device memory only.
    tol
        Fraction of a pixel's peak below which its response counts as zero,
        both when bounding it and when checking it fits its window.
    progress
        Print progress to stderr.

    Returns
    -------
    An EffectiveApertureTable.

    Raises
    ------
    ValueError
        if the optics collects no light, if any pixel is dark, if a
        response reaches the edge of the field (half_angle too small), or if
        one does not fit its window (window too small).
    """
    if frame is None:
        frame = FieldFrame.from_telescope(telescope)
    group = camera.sensor_groups[sensor_idx]

    wavelengths, spectral_area, spectrum, on_axis_area = _bandpass(
        telescope, camera, frame, wavelengths, sensor_idx, progress
    )
    n_pix = int(group.n_sensors * math.prod(group.get_accumulator_shape()))
    cube, axis1d = _field(
        telescope,
        camera,
        frame,
        spectrum,
        sensor_idx,
        n_pix,
        half_angle,
        step,
        chunk_size,
        progress,
    )
    n = axis1d.size
    table_on_axis = float(cube[n // 2, n // 2].sum())
    values, offset, window = _windows(cube, window, tol)
    del cube

    # Re-base so the lowest window corner sits on node 0.
    node = np.full(2, float(step))
    origin = axis1d[0] + offset.min(axis=0) * node
    offset = offset - offset.min(axis=0)

    filled = values > tol * values.max(axis=(1, 2), keepdims=True)
    meta = {
        "telescope": telescope.name,
        "sensor_group": type(group).__name__,
        "n_sensors": int(group.n_sensors),
        "pixel_shape": tuple(int(s) for s in group.get_accumulator_shape()),
        "sensor_idx": int(sensor_idx),
        "half_angle_rad": float(axis1d[-1]),
        "n_directions": n * n,
        "window_fill": float(filled.mean()),
        "n_samples": int(telescope.stage(0).n_samples),
        "on_axis_area_table": table_on_axis,
    }
    return EffectiveApertureTable(
        origin=origin,
        step=node,
        offset=offset,
        values=values,
        on_axis_area=on_axis_area,
        wavelengths=wavelengths,
        spectral_area=spectral_area,
        meta=meta,
    )
