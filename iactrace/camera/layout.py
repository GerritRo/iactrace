"""Pixel layout definitions for camera sensor groups.

Contains the SensorGroup ABC and concrete implementations for square
and hexagonal pixel geometries.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

# SensorGroup ABC


class SensorGroup(eqx.Module):
    """Abstract base class for sensor groups.

    A sensor group contains N sensors at different positions/orientations
    that share the same pixel geometry.

    Attributes:
        positions: Sensor positions in 3D space (N, 3)
        rotations: Sensor rotations as Euler angles in degrees (N, 3)
    """

    positions: Array
    rotations: Array

    @property
    def n_sensors(self) -> int:
        """Return number of sensors in the group."""
        return self.positions.shape[0]

    def __len__(self) -> int:
        return self.n_sensors

    @abstractmethod
    def get_accumulator_shape(self) -> tuple[int, ...]:
        """Return the shape of the accumulator array per sensor."""
        raise NotImplementedError

    @abstractmethod
    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        """Accumulate values at given positions into pixels across all sensors."""
        raise NotImplementedError

    @abstractmethod
    def assign_pixels(
        self, sensor_idx: Array, x: Array, y: Array
    ) -> Array:
        """Assign each ray to a pixel index.

        Args:
            sensor_idx: Index of sensor each ray hit (n_rays,)
            x: X coordinates in local sensor planes (n_rays,)
            y: Y coordinates in local sensor planes (n_rays,)

        Returns:
            Flat pixel indices (n_rays,). Invalid rays get index 0.
        """
        raise NotImplementedError

    @abstractmethod
    def in_bounds(self, x: Array, y: Array) -> Array:
        """Predicate: True for ``(x, y)`` inside the sensor's active footprint.

        Coordinates are in a single sensor's local frame. Used by
        :func:`iactrace.camera.camera.intersect_sensor` to mask rays whose
        plane intersection falls outside this tile's pixel region before
        selecting the closest tile across a multi-sensor group.
        """
        raise NotImplementedError


# Square pixel helpers


def _square_edge_distance(x_frac: Array, y_frac: Array, dx: float, dy: float) -> Array:
    """Compute distance to nearest pixel edge in physical units."""
    dist_to_edge_x = jnp.minimum(x_frac, 1 - x_frac) * dx
    dist_to_edge_y = jnp.minimum(y_frac, 1 - y_frac) * dy
    return jnp.minimum(dist_to_edge_x, dist_to_edge_y)


# SquareSensorGroup


class SquareSensorGroup(SensorGroup):
    """Square pixel sensor group.

    Contains N sensors at different positions/orientations that share the same
    pixel geometry (width, height, bounds). Output shape is (N, height, width).
    """

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
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

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

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) pixel bounds."""
        return (self.x0, self.x0 + self.dx * self.width, self.y0, self.y0 + self.dy * self.height)

    def get_accumulator_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
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

        flat_idx = sensor_idx * (self.height * self.width) + yi * self.width + xi

        values_masked = jnp.where(valid, values, 0.0)
        img_flat = jax.ops.segment_sum(
            values_masked, flat_idx,
            num_segments=self.n_sensors * self.height * self.width
        )

        return img_flat.reshape(self.n_sensors, self.height, self.width)

    def assign_pixels(
        self, sensor_idx: Array, x: Array, y: Array
    ) -> Array:
        x_cont = (x - self.x0) / self.dx
        y_cont = (y - self.y0) / self.dy

        xi = jnp.floor(x_cont).astype(jnp.int32)
        yi = jnp.floor(y_cont).astype(jnp.int32)

        xi = jnp.clip(xi, 0, self.width - 1)
        yi = jnp.clip(yi, 0, self.height - 1)

        return sensor_idx * (self.height * self.width) + yi * self.width + xi

    def in_bounds(self, x: Array, y: Array) -> Array:
        x_max = self.x0 + self.dx * self.width
        y_max = self.y0 + self.dy * self.height
        return (x >= self.x0) & (x <= x_max) & (y >= self.y0) & (y <= y_max)


# Hexagonal pixel helpers

SQRT3: float = 1.7320508075688772
SQRT3_2: float = 0.8660254037844386   # sqrt(3)/2
SQRT3_3: float = 0.5773502691896257   # 1/sqrt(3)


def _rotate(x: Array, y: Array, angle: float | Array) -> tuple[Array, Array]:
    """Rotate 2D coordinates by angle."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return c * x - s * y, s * x + c * y


def _cartesian_to_axial(x: Array, y: Array, size: float) -> tuple[Array, Array]:
    """Cartesian to axial hex coordinates (pointy-top)."""
    return (SQRT3_3 * x - y / 3) / size, (2 * y / 3) / size


def _axial_to_cartesian(q: Array, r: Array, size: float) -> tuple[Array, Array]:
    """Axial to Cartesian hex coordinates (pointy-top)."""
    return size * SQRT3 * (q + r / 2), size * 1.5 * r


def _axial_round(q: Array, r: Array) -> tuple[Array, Array]:
    """Round fractional axial coordinates to nearest hex center."""
    s = -q - r
    qi, ri, si = jnp.round(q), jnp.round(r), jnp.round(s)
    dq, dr, ds = jnp.abs(qi - q), jnp.abs(ri - r), jnp.abs(si - s)
    qi = jnp.where((dq > dr) & (dq > ds), -ri - si, qi)
    ri = jnp.where((dr > dq) & (dr > ds), -qi - si, ri)
    return qi, ri


def _hex_norm(x: Array, y: Array, inradius: float) -> Array:
    """Hexagonal norm: 0 at center, 1 at boundary (pointy-top)."""
    return jnp.maximum(jnp.abs(x), 0.5 * jnp.abs(x) + SQRT3_2 * jnp.abs(y)) / inradius


def _detect_hex_grid(centers: Array) -> tuple[Array, Array, Array]:
    """Detect hex size, rotation, and offset from center positions."""
    centers = jnp.asarray(centers)
    n = len(centers)

    diff = centers[:, None] - centers[None, :]
    dist_sq = jnp.sum(diff**2, axis=2)
    dist_sq = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist_sq)
    min_dist = jnp.sqrt(jnp.min(dist_sq))

    idx = jnp.argmin(dist_sq)
    vec = diff[idx // n, idx % n]
    angle = jnp.mod(jnp.arctan2(vec[1], vec[0]), jnp.pi / 3)

    offset = centers[jnp.argmin(jnp.sum(centers**2, axis=1))]

    return min_dist / SQRT3, angle, offset


def _build_lookup_table(
    centers: Array, hex_size: float, rotation: float, offset: Array
) -> tuple[Array, int, int]:
    """Build axial coordinate lookup table from hex centers."""
    x = centers[:, 0] - offset[0]
    y = centers[:, 1] - offset[1]
    x_rot, y_rot = _rotate(x, y, -rotation)

    q, r = _cartesian_to_axial(x_rot, y_rot, hex_size)
    qi = jnp.round(q).astype(jnp.int32)
    ri = jnp.round(r).astype(jnp.int32)

    q_min, q_max = int(qi.min()), int(qi.max())
    r_min, r_max = int(ri.min()), int(ri.max())

    table = jnp.full((q_max - q_min + 1, r_max - r_min + 1), -1, dtype=jnp.int32)
    table = table.at[qi - jnp.int32(q_min), ri - jnp.int32(r_min)].set(
        jnp.arange(len(centers), dtype=jnp.int32)
    )

    return table, q_min, r_min


# HexagonalSensorGroup


class HexagonalSensorGroup(SensorGroup):
    """Hexagonal pixel sensor group.

    Contains N sensors at different positions/orientations that share the same
    hexagonal pixel geometry. Output shape is (N, n_pixels).
    """

    positions: Array
    rotations: Array
    hex_centers: Array
    lookup_table: Array

    hex_size: float = eqx.field(static=True)
    hex_inradius: float = eqx.field(static=True)
    grid_rotation: float = eqx.field(static=True)
    grid_offset: tuple[float, float] = eqx.field(static=True)
    q_min: int = eqx.field(static=True)
    r_min: int = eqx.field(static=True)
    n_pixels: int = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | Array,
        rotations: Sequence[Sequence[float]] | Array,
        hex_centers: Sequence[Sequence[float]] | Array,
        edge_width: float = 0.0,
    ) -> None:
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if self.rotations.ndim == 1:
            self.rotations = self.rotations[None, :]

        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        self.edge_width = float(edge_width)

        size, rot, offset = _detect_hex_grid(self.hex_centers)
        self.hex_size = float(size)
        self.hex_inradius = float(size * SQRT3_2)
        self.grid_rotation = float(rot)
        self.grid_offset = (float(offset[0]), float(offset[1]))

        self.lookup_table, self.q_min, self.r_min = _build_lookup_table(
            self.hex_centers, self.hex_size, self.grid_rotation, offset
        )

    def get_accumulator_shape(self) -> tuple[int]:
        return (self.n_pixels,)

    def _to_grid_coords(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Transform world coordinates to grid-aligned coordinates."""
        return _rotate(
            x - self.grid_offset[0], y - self.grid_offset[1], -self.grid_rotation
        )

    def _lookup_pixels(self, qi: Array, ri: Array) -> tuple[Array, Array]:
        """Look up pixel indices from axial coordinates, handling bounds."""
        q_idx = qi - jnp.int32(self.q_min)
        r_idx = ri - jnp.int32(self.r_min)

        in_bounds = (
            (q_idx >= 0)
            & (q_idx < self.lookup_table.shape[0])
            & (r_idx >= 0)
            & (r_idx < self.lookup_table.shape[1])
        )

        q_safe = jnp.clip(q_idx, 0, self.lookup_table.shape[0] - 1)
        r_safe = jnp.clip(r_idx, 0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_safe, r_safe]

        valid = in_bounds & (pixel_idx >= 0)
        return jnp.where(valid, pixel_idx, 0), valid

    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)

        pixel_idx, valid = self._lookup_pixels(
            qi.astype(jnp.int32), ri.astype(jnp.int32)
        )

        hex_center_x, hex_center_y = _axial_to_cartesian(qi, ri, self.hex_size)
        hex_dist = _hex_norm(
            x_grid - hex_center_x, y_grid - hex_center_y, self.hex_inradius
        )
        edge_threshold = 1.0 - self.edge_width / self.hex_inradius
        on_edge = hex_dist > edge_threshold
        valid = valid & ~on_edge

        flat_idx = sensor_idx * self.n_pixels + pixel_idx

        result = jax.ops.segment_sum(
            jnp.where(valid, values, 0.0), flat_idx,
            num_segments=self.n_sensors * self.n_pixels
        )
        return result.reshape(self.n_sensors, self.n_pixels)

    def assign_pixels(
        self, sensor_idx: Array, x: Array, y: Array
    ) -> Array:
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)

        pixel_idx, _valid = self._lookup_pixels(
            qi.astype(jnp.int32), ri.astype(jnp.int32)
        )

        return sensor_idx * self.n_pixels + pixel_idx

    def in_bounds(self, x: Array, y: Array) -> Array:
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)
        _, valid = self._lookup_pixels(qi.astype(jnp.int32), ri.astype(jnp.int32))
        return valid