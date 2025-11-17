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
            orientation=0,  # not flat-top
            facecolor=cmap((value - vmin) / (vmax - vmin)) if vmax > vmin else 'white',
            edgecolor='gray',
            linewidth=0.5,
            **kwargs
        )
        ax.add_patch(hexagon)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax