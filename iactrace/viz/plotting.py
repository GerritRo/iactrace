import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


def squareshow(result, sensor, ax=None, **kwargs):
    """
    Display square pixel data.

    Args:
        result: Grid for square sensor
        sensor: square sensor
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional arguments for square plotting (vmin, vmax, cmap, etc.)

    Returns:
        Matplotlib axis
    """
    if ax is None:
        ax = plt.gca()

    vmin = kwargs.pop('vmin', result.min())
    vmax = kwargs.pop('vmax', result.max())
    cmap = kwargs.pop('cmap', plt.cm.viridis)

    x_extent = sensor.x0 + sensor.width * sensor.dx
    y_extent = sensor.y0 + sensor.height * sensor.dy
    ax.imshow(result, origin='lower', extent=[sensor.x0, x_extent, sensor.y0, y_extent],
              vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax


def hexshow(result, sensor, ax=None, **kwargs):
    """
    Display hexagonal pixel data.

    Args:
        result: Values for each hexagon (N,)
        sensor: Hexagonal sensor
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional arguments for hexagon plotting (vmin, vmax, cmap, etc.)

    Returns:
        Matplotlib axis
    """
    if ax is None:
        ax = plt.gca()

    hex_centers = np.array(sensor.hex_centers)
    
    # Determine hex size for plotting
    hex_size = sensor.hex_size
    
    # Get the grid rotation (how much the original grid was rotated)
    # We need to counter-rotate the hexagon orientation
    grid_rotation = -sensor.rotation_angle  # Negative because we counter-rotate

    vmin = kwargs.pop('vmin', result.min())
    vmax = kwargs.pop('vmax', result.max())
    cmap = kwargs.pop('cmap', plt.cm.viridis)

    for i, (x, y) in enumerate(hex_centers):
        value = float(result[i])
        color = plt.cm.viridis(value / vmax)
        
        # Pointy-top base orientation (30 degrees) plus the grid's rotation
        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_size,
            orientation=grid_rotation,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(hex_patch)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax