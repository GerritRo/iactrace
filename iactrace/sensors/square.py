from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .base import SensorGroup


def _bilinear_splat_forward(
    sensor_idx: Array,
    x: Array,
    y: Array,
    values: Array,
    n_sensors: int,
    width: int,
    height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    edge_width: float,
) -> Array:
    """Forward pass: hard assignment (identical to SquareSensorGroup)."""
    x_cont = (x - x0) / dx
    y_cont = (y - y0) / dy

    xi = jnp.floor(x_cont).astype(jnp.int32)
    yi = jnp.floor(y_cont).astype(jnp.int32)

    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    x_frac = x_cont - xi
    y_frac = y_cont - yi
    dist_to_edge = _square_edge_distance(x_frac, y_frac, dx, dy)
    on_edge = dist_to_edge < edge_width
    valid = valid & ~on_edge

    xi = jnp.clip(xi, 0, width - 1)
    yi = jnp.clip(yi, 0, height - 1)

    # 3D flat index: sensor_idx * (height * width) + yi * width + xi
    flat_idx = sensor_idx * (height * width) + yi * width + xi

    values_masked = jnp.where(valid, values, 0.0)
    img_flat = jax.ops.segment_sum(
        values_masked, flat_idx, num_segments=n_sensors * height * width
    )

    return img_flat.reshape(n_sensors, height, width)


def _bilinear_splat_backward(
    sensor_idx: Array,
    x: Array,
    y: Array,
    values: Array,
    g: Array,
    n_sensors: int,
    width: int,
    height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    edge_width: float,
) -> tuple[Array, Array, Array]:
    """Backward pass: bilinear interpolation gradients.

    Computes gradients as if the forward pass used bilinear splatting to 4 neighbors.
    """
    x_cont = (x - x0) / dx
    y_cont = (y - y0) / dy

    xi = jnp.floor(x_cont).astype(jnp.int32)
    yi = jnp.floor(y_cont).astype(jnp.int32)

    x_frac = x_cont - xi
    y_frac = y_cont - yi

    # Check validity for all 4 bilinear neighbors
    valid_base = (xi >= 0) & (xi < width - 1) & (yi >= 0) & (yi < height - 1)

    # Edge exclusion (applied to center position)
    dist_to_edge = _square_edge_distance(x_frac, y_frac, dx, dy)
    on_edge = dist_to_edge < edge_width
    valid = valid_base & ~on_edge

    # Bilinear weights for 4 neighbors: (xi, yi), (xi+1, yi), (xi, yi+1), (xi+1, yi+1)
    w00 = (1 - x_frac) * (1 - y_frac)  # (xi, yi)
    w10 = x_frac * (1 - y_frac)  # (xi+1, yi)
    w01 = (1 - x_frac) * y_frac  # (xi, yi+1)
    w11 = x_frac * y_frac  # (xi+1, yi+1)

    # Clamp indices for safe access
    xi_safe = jnp.clip(xi, 0, width - 2)
    yi_safe = jnp.clip(yi, 0, height - 2)

    # Get output gradients at each of the 4 neighbor pixels
    # Use 3D indexing: g[sensor_idx, yi, xi]
    g_flat = g.reshape(-1)
    pixels_per_sensor = height * width
    base_idx = sensor_idx * pixels_per_sensor

    idx00 = base_idx + yi_safe * width + xi_safe
    idx10 = base_idx + yi_safe * width + (xi_safe + 1)
    idx01 = base_idx + (yi_safe + 1) * width + xi_safe
    idx11 = base_idx + (yi_safe + 1) * width + (xi_safe + 1)

    g00 = g_flat[idx00]
    g10 = g_flat[idx10]
    g01 = g_flat[idx01]
    g11 = g_flat[idx11]

    # Gradient w.r.t. values: sum of weighted output gradients
    grad_values = jnp.where(valid, w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11, 0.0)

    # Gradient w.r.t. x_frac: d/d(x_frac) of bilinear weights
    dw_dxf_00 = -(1 - y_frac)
    dw_dxf_10 = 1 - y_frac
    dw_dxf_01 = -y_frac
    dw_dxf_11 = y_frac

    # Gradient w.r.t. y_frac
    dw_dyf_00 = -(1 - x_frac)
    dw_dyf_10 = -x_frac
    dw_dyf_01 = 1 - x_frac
    dw_dyf_11 = x_frac

    # Chain rule: grad_x_frac = values * sum(d(weight)/d(x_frac) * g_pixel)
    grad_x_frac = values * (
        dw_dxf_00 * g00 + dw_dxf_10 * g10 + dw_dxf_01 * g01 + dw_dxf_11 * g11
    )
    grad_y_frac = values * (
        dw_dyf_00 * g00 + dw_dyf_10 * g10 + dw_dyf_01 * g01 + dw_dyf_11 * g11
    )

    # Convert from fractional coords to world coords: x_frac = (x - x0) / dx - floor(...)
    # d(x_frac)/d(x) = 1/dx (the floor doesn't contribute gradient)
    grad_x = jnp.where(valid, grad_x_frac / dx, 0.0)
    grad_y = jnp.where(valid, grad_y_frac / dy, 0.0)

    return grad_x, grad_y, grad_values


def _make_ste_square_accumulate(
    n_sensors: int,
    width: int,
    height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    edge_width: float,
):
    """Create a straight-through estimator accumulate function with custom_vjp."""

    @jax.custom_vjp
    def ste_accumulate(
        sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        return _bilinear_splat_forward(
            sensor_idx, x, y, values, n_sensors, width, height, x0, y0, dx, dy, edge_width
        )

    def ste_accumulate_fwd(sensor_idx: Array, x: Array, y: Array, values: Array):
        result = _bilinear_splat_forward(
            sensor_idx, x, y, values, n_sensors, width, height, x0, y0, dx, dy, edge_width
        )
        return result, (sensor_idx, x, y, values)

    def ste_accumulate_bwd(residuals, g):
        sensor_idx, x, y, values = residuals
        grad_x, grad_y, grad_values = _bilinear_splat_backward(
            sensor_idx, x, y, values, g, n_sensors, width, height, x0, y0, dx, dy, edge_width
        )
        # No gradient for sensor_idx (it's an integer index)
        return None, grad_x, grad_y, grad_values

    ste_accumulate.defvjp(ste_accumulate_fwd, ste_accumulate_bwd)
    return ste_accumulate


def _square_edge_distance(x_frac: Array, y_frac: Array, dx: float, dy: float) -> Array:
    """Compute distance to nearest pixel edge.

    Args:
        x_frac: Fractional x position within pixel (0 to 1)
        y_frac: Fractional y position within pixel (0 to 1)
        dx: Pixel width in physical units
        dy: Pixel height in physical units

    Returns:
        Distance to nearest edge in physical units
    """
    dist_to_edge_x = jnp.minimum(x_frac, 1 - x_frac) * dx
    dist_to_edge_y = jnp.minimum(y_frac, 1 - y_frac) * dy
    return jnp.minimum(dist_to_edge_x, dist_to_edge_y)


class SquareSensorGroup(SensorGroup):
    """Square pixel sensor group.

    Contains N sensors at different positions/orientations that share the same
    pixel geometry (width, height, bounds). Output shape is (N, height, width).
    """

    config_type: ClassVar[str] = "square"

    positions: Array
    rotations: Array

    width: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    x0: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | Array,
        rotations: Sequence[Sequence[float]] | Array,
        width: int,
        height: int,
        bounds: tuple[float, float, float, float],
        edge_width: float = 0.0,
    ) -> None:
        """Initialize square sensor group.

        Args:
            positions: Sensor positions (N, 3) in world coordinates
            rotations: Sensor rotations (N, 3) as Euler angles in degrees
            width: Number of pixels in x direction (shared by all sensors)
            height: Number of pixels in y direction (shared by all sensors)
            bounds: (x_min, x_max, y_min, y_max) pixel bounds (shared by all sensors)
            edge_width: Width of edge exclusion zone (shared by all sensors)
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

        # Ensure 2D arrays even for single sensor
        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if self.rotations.ndim == 1:
            self.rotations = self.rotations[None, :]

        self.width = int(width)
        self.height = int(height)
        self.edge_width = float(edge_width)

        xmin, xmax, ymin, ymax = bounds
        self.x0 = float(xmin)
        self.y0 = float(ymin)
        self.dx = float((xmax - xmin) / width)
        self.dy = float((ymax - ymin) / height)

    def get_accumulator_shape(self) -> tuple[int, int]:
        """Return per-sensor accumulator shape: (height, width)."""
        return (self.height, self.width)

    def to_config(self, index: int = 0) -> dict[str, Any]:
        """Convert sensor at index to config dict."""
        x_max = self.x0 + self.dx * self.width
        y_max = self.y0 + self.dy * self.height
        config: dict[str, Any] = {
            "type": "square",
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "width": self.width,
            "height": self.height,
            "bounds": [self.x0, x_max, self.y0, y_max],
        }
        if self.edge_width > 0:
            config["edge_width"] = self.edge_width
        return config

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> SquareSensorGroup:
        """Create SquareSensorGroup from list of config dicts.

        All sensors must share the same pixel geometry (width, height, bounds).
        """
        if not configs:
            raise ValueError("At least one sensor config is required")

        # Use first config to get shared geometry
        first = configs[0]
        bounds = first["bounds"]
        width = first["width"]
        height = first["height"]
        edge_width = first.get("edge_width", 0.0)

        # Collect positions and rotations
        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]

        return cls(
            positions=positions,
            rotations=rotations,
            width=width,
            height=height,
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            edge_width=edge_width,
        )

    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        """Accumulate photon hits into pixels across all sensors.

        Args:
            sensor_idx: Index of sensor each ray hit (n_rays,)
            x: X coordinates in local sensor planes (n_rays,)
            y: Y coordinates in local sensor planes (n_rays,)
            values: Values to accumulate (n_rays,)

        Returns:
            Accumulated images (n_sensors, height, width)
        """
        x_cont = (x - self.x0) / self.dx
        y_cont = (y - self.y0) / self.dy

        xi = jnp.floor(x_cont).astype(jnp.int32)
        yi = jnp.floor(y_cont).astype(jnp.int32)

        valid = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)

        x_frac = x_cont - xi
        y_frac = y_cont - yi
        dist_to_edge = _square_edge_distance(x_frac, y_frac, self.dx, self.dy)
        on_edge = dist_to_edge < self.edge_width
        valid = valid & ~on_edge

        xi = jnp.clip(xi, 0, self.width - 1)
        yi = jnp.clip(yi, 0, self.height - 1)

        # 3D flat index: sensor_idx * (height * width) + yi * width + xi
        flat_idx = sensor_idx * (self.height * self.width) + yi * self.width + xi

        values_masked = jnp.where(valid, values, 0.0)
        img_flat = jax.ops.segment_sum(
            values_masked, flat_idx,
            num_segments=self.n_sensors * self.height * self.width
        )

        return img_flat.reshape(self.n_sensors, self.height, self.width)


class StraightThroughSquareSensorGroup(SensorGroup):
    """Square sensor group with straight-through estimator.

    Forward pass uses hard assignment (like SquareSensorGroup).
    Backward pass uses bilinear interpolation for gradient computation.

    Note: This is not registered with sensor_registry because YAML configs
    always load as SquareSensorGroup. Use telescope.with_ste() to convert.
    """

    # Use same config_type as SquareSensorGroup for serialization compatibility
    config_type = "square"

    positions: Array
    rotations: Array

    width: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    x0: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | Array,
        rotations: Sequence[Sequence[float]] | Array,
        width: int,
        height: int,
        bounds: tuple[float, float, float, float],
        edge_width: float = 0.0,
    ) -> None:
        """Initialize straight-through square sensor group.

        Args:
            positions: Sensor positions (N, 3) in world coordinates
            rotations: Sensor rotations (N, 3) as Euler angles in degrees
            width: Number of pixels in x direction
            height: Number of pixels in y direction
            bounds: (x_min, x_max, y_min, y_max) pixel bounds
            edge_width: Width of edge exclusion zone
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

        # Ensure 2D arrays even for single sensor
        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if self.rotations.ndim == 1:
            self.rotations = self.rotations[None, :]

        self.width = int(width)
        self.height = int(height)
        self.edge_width = float(edge_width)

        xmin, xmax, ymin, ymax = bounds
        self.x0 = float(xmin)
        self.y0 = float(ymin)
        self.dx = float((xmax - xmin) / width)
        self.dy = float((ymax - ymin) / height)

    def get_accumulator_shape(self) -> tuple[int, int]:
        """Return per-sensor accumulator shape: (height, width)."""
        return (self.height, self.width)

    def to_config(self, index: int = 0) -> dict[str, Any]:
        """Convert sensor at index to config dict."""
        x_max = self.x0 + self.dx * self.width
        y_max = self.y0 + self.dy * self.height
        config: dict[str, Any] = {
            "type": "square",
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "width": self.width,
            "height": self.height,
            "bounds": [self.x0, x_max, self.y0, y_max],
        }
        if self.edge_width > 0:
            config["edge_width"] = self.edge_width
        return config

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> StraightThroughSquareSensorGroup:
        """Create StraightThroughSquareSensorGroup from list of config dicts."""
        if not configs:
            raise ValueError("At least one sensor config is required")

        first = configs[0]
        bounds = first["bounds"]
        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]

        return cls(
            positions=positions,
            rotations=rotations,
            width=first["width"],
            height=first["height"],
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            edge_width=first.get("edge_width", 0.0),
        )

    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        """Accumulate with straight-through estimator.

        Forward: hard assignment to single pixel.
        Backward: bilinear interpolation gradients.
        """
        ste_fn = _make_ste_square_accumulate(
            self.n_sensors,
            self.width,
            self.height,
            self.x0,
            self.y0,
            self.dx,
            self.dy,
            self.edge_width,
        )
        return ste_fn(sensor_idx, x, y, values)
