from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from ..core.transforms import euler_to_matrix


def show_camera(
    image,
    sensor,
    ax=None,
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    norm=None,
    edgecolor="none",
    linewidth=0.0,
    colorbar=False,
    cbar_label=None,
):
    """Render a camera image at actual focal-plane positions.

    Builds a single :class:`~matplotlib.collections.PolyCollection` holding
    every pixel in the camera. Pixel polygons are projected onto the camera
    ``(x, y)`` plane after applying each tile's position and Euler rotation,
    so curvature and tile tilt are reflected faithfully.

    Args:
        image: Pixel image array. Shape ``(n_sensors, height, width)`` for
            :class:`~iactrace.SquareSensorGroup`, ``(n_sensors, n_pixels)``
            for :class:`~iactrace.HexagonalSensorGroup`.
        sensor: The sensor group that produced ``image``.
        ax: Matplotlib axes. Created with a square aspect figure if ``None``.
        cmap: Colormap name or instance.
        vmin, vmax: Explicit colour limits. Default: data min/max.
        norm: Optional :class:`matplotlib.colors.Normalize`. Overrides
            ``vmin``/``vmax`` when given.
        edgecolor: Pixel-boundary colour. Default ``"none"`` (no outlines).
        linewidth: Pixel-boundary line width.
        colorbar: Attach a vertical colorbar to ``ax``.
        cbar_label: Colorbar label text.

    Returns:
        ``ax`` (the axes that received the collection).
    """
    from ..camera.sensor_group import HexagonalSensorGroup, SquareSensorGroup

    image = np.asarray(image)
    if isinstance(sensor, SquareSensorGroup):
        polys, values = _square_polygons(image, sensor)
    elif isinstance(sensor, HexagonalSensorGroup):
        polys, values = _hex_polygons(image, sensor)
    else:
        raise TypeError(f"show_camera does not support sensor type {type(sensor).__name__}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if norm is None:
        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))

    pc = PolyCollection(
        polys,
        array=values,
        cmap=cmap,
        norm=norm,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    if norm is None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.autoscale_view()

    if colorbar:
        cb = ax.get_figure().colorbar(pc, ax=ax)
        if cbar_label is not None:
            cb.set_label(cbar_label)
    return ax


def _stack_rotations(rotations):
    """Per-sensor rotation matrices, shape (n_sensors, 3, 3).

    Uses the canonical :func:`iactrace.core.transforms.euler_to_matrix` so the
    2D camera view shares the exact Euler convention of the 3D geometry.
    """
    return np.stack([np.asarray(euler_to_matrix(r)) for r in np.asarray(rotations)])


def _square_polygons(image, sensor):
    """Return ``(polys, values)`` for a SquareSensorGroup image."""
    expected_shape = (sensor.n_sensors, sensor.height, sensor.width)
    if image.shape != expected_shape:
        raise ValueError(f"image shape {image.shape} does not match sensor shape {expected_shape}")

    h, w = sensor.height, sensor.width
    xs = sensor.x0 + np.arange(w + 1) * sensor.dx  # (w+1,)
    ys = sensor.y0 + np.arange(h + 1) * sensor.dy  # (h+1,)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")  # (h+1, w+1)
    # CCW pixel quads: BL, BR, TR, TL.
    corners_2d = np.stack(
        [
            np.stack([XX[:-1, :-1], YY[:-1, :-1]], axis=-1),
            np.stack([XX[:-1, 1:], YY[:-1, 1:]], axis=-1),
            np.stack([XX[1:, 1:], YY[1:, 1:]], axis=-1),
            np.stack([XX[1:, :-1], YY[1:, :-1]], axis=-1),
        ],
        axis=2,
    )  # (h, w, 4, 2)
    z = np.zeros(corners_2d.shape[:-1] + (1,))
    local = np.concatenate([corners_2d, z], axis=-1)  # (h, w, 4, 3)

    Rs = _stack_rotations(sensor.rotations)  # (n, 3, 3)
    pos = np.asarray(sensor.positions)  # (n, 3)
    world = np.einsum("sij,hwcj->shwci", Rs, local) + pos[:, None, None, None, :]  # (n, h, w, 4, 3)

    polys = world[..., :2].reshape(-1, 4, 2)
    values = np.asarray(image).reshape(-1)
    return polys, values


def _hex_polygons(image, sensor):
    """Return ``(polys, values)`` for a HexagonalSensorGroup image."""
    expected_shape = (sensor.n_sensors, sensor.n_pixels)
    if image.shape != expected_shape:
        raise ValueError(f"image shape {image.shape} does not match sensor shape {expected_shape}")

    centers = np.asarray(sensor.hex_centers)  # (n_pix, 2)
    s = sensor.hex_size
    angles = np.deg2rad(np.arange(30.0, 360.0, 60.0)) + sensor.grid_rotation
    vertex_offsets = s * np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (6, 2)

    pix_corners_2d = centers[:, None, :] + vertex_offsets[None, :, :]  # (n_pix, 6, 2)
    z = np.zeros(pix_corners_2d.shape[:-1] + (1,))
    local = np.concatenate([pix_corners_2d, z], axis=-1)  # (n_pix, 6, 3)

    Rs = _stack_rotations(sensor.rotations)
    pos = np.asarray(sensor.positions)
    world = np.einsum("sij,pcj->spci", Rs, local) + pos[:, None, None, :]  # (n, n_pix, 6, 3)

    polys = world[..., :2].reshape(-1, 6, 2)
    values = np.asarray(image).reshape(-1)
    return polys, values
