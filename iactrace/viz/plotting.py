"""Visualization utilities for telescope simulations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


def hexshow(result, hex_centers, hex_size=None, ax=None, **kwargs):
    """
    Display hexagonal pixel data.

    Args:
        result: Values for each hexagon (N,)
        hex_centers: Hexagon center positions (N, 2)
        hex_size: Hex size (auto-computed if None)
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional arguments for hexagon plotting (vmin, vmax, cmap, etc.)

    Returns:
        Matplotlib axis
    """
    if ax is None:
        ax = plt.gca()

    # Auto-compute hex size if not provided
    if hex_size is None:
        diff = hex_centers[:, None, :] - hex_centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        hex_size = min_dist / np.sqrt(3)

    vmin = kwargs.pop('vmin', result.min())
    vmax = kwargs.pop('vmax', result.max())
    cmap = kwargs.pop('cmap', plt.cm.viridis)

    for i, (x, y) in enumerate(hex_centers):
        value = result[i]

        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_size,
            orientation=np.pi/6,  # flat-top
            facecolor=cmap((value - vmin) / (vmax - vmin)) if vmax > vmin else 'white',
            edgecolor='gray',
            linewidth=0.5,
            **kwargs
        )
        ax.add_patch(hexagon)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax


def plot_telescope_geometry(telescope, ax=None):
    """
    Plot telescope mirror layout (top-down view).

    Args:
        telescope: Telescope object
        ax: Matplotlib axis (creates new if None)

    Returns:
        Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot mirrors
    positions = telescope.mirror_positions[:, :2]
    if telescope.mirror_positions.shape[1] > 2:
        radii = telescope.mirror_positions[:, 2] / 2
    else:
        radii = np.full(len(positions), 0.3)

    for pos, r in zip(positions, radii):
        circle = plt.Circle(pos, r, fill=False, edgecolor='blue', linewidth=1.5)
        ax.add_patch(circle)

    # Plot obstructions
    from ..telescope.obstructions import Cylinder, Box

    for obs in telescope.obstructions:
        if isinstance(obs, Cylinder):
            # Draw cylinder as line with width
            p1, p2 = np.array(obs.p1), np.array(obs.p2)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=obs.radius*100, alpha=0.6)
        elif isinstance(obs, Box):
            # Draw box as rectangle (top-down view shows x-y projection)
            p1, p2 = np.array(obs.p1), np.array(obs.p2)
            box_min = np.minimum(p1, p2)
            box_max = np.maximum(p1, p2)
            width = box_max[0] - box_min[0]
            height = box_max[1] - box_min[1]
            rect = plt.Rectangle(
                (box_min[0], box_min[1]),
                width, height,
                fill=True, facecolor='green', alpha=0.3,
                edgecolor='green', linewidth=2
            )
            ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Telescope Mirror Layout')
    ax.grid(True, alpha=0.3)

    return ax


def plot_focal_plane(image, sensor_config, ax=None, **kwargs):
    """
    Plot focal plane image.

    Args:
        image: Rendered image
        sensor_config: Sensor configuration dict
        ax: Matplotlib axis
        **kwargs: Additional arguments for imshow/hexshow

    Returns:
        Matplotlib axis
    """
    if ax is None:
        ax = plt.gca()

    if sensor_config['type'] == 'square':
        im = ax.imshow(image, origin='lower', extent=[
            sensor_config['x0'],
            sensor_config['x0'] + sensor_config['width'] * sensor_config['dx'],
            sensor_config['y0'],
            sensor_config['y0'] + sensor_config['height'] * sensor_config['dy']
        ], **kwargs)
        plt.colorbar(im, ax=ax)
    elif sensor_config['type'] == 'hexagonal':
        hexshow(image, sensor_config['hex_centers'], ax=ax, **kwargs)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Focal Plane')

    return ax
