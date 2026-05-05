import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize


def squareshow(result, sensor, sensor_idx=None, ax=None, **kwargs):
    """
    Display square pixel data from a sensor group.

    Args:
        result: Accumulated result with shape (n_sensors, height, width)
        sensor: SquareSensorGroup
        sensor_idx: Index of sensor to display. If None:
            - For single-sensor groups (n_sensors=1): displays that sensor
            - For multi-sensor groups: defaults to grid visualization (calls show_sensor_grid)
        ax: Matplotlib axis (creates new if None). If provided for multi-sensor groups,
            sensor_idx must be specified.
        **kwargs: Additional arguments for square plotting (vmin, vmax, cmap, etc.)

    Returns:
        Matplotlib axis (if single sensor or sensor_idx specified)
        OR tuple of (Figure, array of axes) (if multi-sensor with no sensor_idx)
    """
    # Handle sensor group output shape (n_sensors, height, width)
    if result.ndim == 3:
        n_sensors = result.shape[0]
        if sensor_idx is not None:
            data = result[sensor_idx]
        elif n_sensors == 1:
            data = result[0]
        else:
            # Multi-sensor case with no sensor_idx specified
            if ax is not None:
                raise ValueError(
                    "For multi-sensor groups, either specify sensor_idx to display a single sensor, "
                    "or omit the ax parameter to use grid visualization."
                )
            # Default to grid visualization for multi-sensor groups
            return show_sensor_grid(result, sensor, **kwargs)
    else:
        # Backward compatibility: accept 2D input
        data = result

    if ax is None:
        ax = plt.gca()

    vmin = kwargs.pop('vmin', data.min())
    vmax = kwargs.pop('vmax', data.max())
    cmap = plt.get_cmap(kwargs.pop('cmap', 'viridis'))

    x_extent = sensor.x0 + sensor.width * sensor.dx
    y_extent = sensor.y0 + sensor.height * sensor.dy
    ax.imshow(data, origin='lower', extent=[sensor.x0, x_extent, sensor.y0, y_extent],
              vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax


def hexshow(result, sensor, sensor_idx=None, ax=None, **kwargs):
    """
    Display hexagonal pixel data from a sensor group.

    Args:
        result: Accumulated result with shape (n_sensors, n_pixels)
        sensor: HexagonalSensorGroup
        sensor_idx: Index of sensor to display. If None:
            - For single-sensor groups (n_sensors=1): displays that sensor
            - For multi-sensor groups: defaults to grid visualization (calls show_sensor_grid)
        ax: Matplotlib axis (creates new if None). If provided for multi-sensor groups,
            sensor_idx must be specified.
        **kwargs: Additional arguments for hexagon plotting (vmin, vmax, cmap, etc.)

    Returns:
        Matplotlib axis (if single sensor or sensor_idx specified)
        OR tuple of (Figure, array of axes) (if multi-sensor with no sensor_idx)
    """
    # Handle sensor group output shape (n_sensors, n_pixels)
    if result.ndim == 2:
        n_sensors = result.shape[0]
        if sensor_idx is not None:
            data = result[sensor_idx]
        elif n_sensors == 1:
            data = result[0]
        else:
            # Multi-sensor case with no sensor_idx specified
            if ax is not None:
                raise ValueError(
                    "For multi-sensor groups, either specify sensor_idx to display a single sensor, "
                    "or omit the ax parameter to use grid visualization."
                )
            # Default to grid visualization for multi-sensor groups
            return show_sensor_grid(result, sensor, **kwargs)
    else:
        # Backward compatibility: accept 1D input
        data = result

    if ax is None:
        ax = plt.gca()

    hex_centers = np.array(sensor.hex_centers)
    hex_size = sensor.hex_size
    grid_rotation = sensor.grid_rotation

    vmin = kwargs.pop('vmin', data.min())
    vmax = kwargs.pop('vmax', data.max())
    cmap = plt.get_cmap(kwargs.pop('cmap', 'viridis'))
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, (x, y) in enumerate(hex_centers):
        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_size,
            orientation=grid_rotation,
            facecolor=cmap(norm(float(data[i]))),
            edgecolor='black',
            linewidth=0.5,
        )
        ax.add_patch(hex_patch)

    ax.set_aspect('equal')
    ax.autoscale_view()
    return ax


def show_sensor_grid(result, sensor, ncols=None, figsize=None, **kwargs):
    """
    Display all sensors in a sensor group as a grid of subplots.

    Args:
        result: Accumulated result with shape (n_sensors, ...)
        sensor: SensorGroup (SquareSensorGroup or HexagonalSensorGroup)
        ncols: Number of columns in grid (default: auto)
        figsize: Figure size tuple (default: auto based on grid size)
        **kwargs: Additional arguments passed to squareshow/hexshow

    Returns:
        Figure and array of axes
    """
    from ..sensors import HexagonalSensorGroup, SquareSensorGroup

    n_sensors = result.shape[0]

    if ncols is None:
        ncols = min(4, n_sensors)
    nrows = (n_sensors + ncols - 1) // ncols

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if n_sensors == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_sensors):
        ax = axes[i]
        if isinstance(sensor, SquareSensorGroup):
            squareshow(result, sensor, sensor_idx=i, ax=ax, **kwargs)
        elif isinstance(sensor, HexagonalSensorGroup):
            hexshow(result, sensor, sensor_idx=i, ax=ax, **kwargs)
        ax.set_title(f"Sensor {i}")

    # Hide unused axes
    for i in range(n_sensors, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes
