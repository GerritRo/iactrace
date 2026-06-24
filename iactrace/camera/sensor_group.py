from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.ray_bundle import RayBundle
from ._hexgeom import (
    SQRT3_2,
    _axial_round,
    _axial_to_cartesian,
    _build_lookup_table,
    _cartesian_to_axial,
    _detect_hex_grid,
    _hex_norm,
    _rotate,
)
from .chain import DetectionChain
from .photosensor import UniformQE

if TYPE_CHECKING:
    from .concentrator import Concentrator
    from .photosensor import PhotoSensor


# Detection-chain helper


def _build_chain(
    concentrator: Concentrator | None,
    photosensor: PhotoSensor | None,
    gap: float,
) -> DetectionChain:
    """Assemble a :class:`DetectionChain`, defaulting to a perfect flat QE.

    ``photosensor=None`` becomes a :class:`~iactrace.camera.photosensor.UniformQE`
    with unit efficiency, so a geometry-only sensor group still detects every
    incident ray.
    """
    return DetectionChain(
        concentrator=concentrator,
        photosensor=photosensor if photosensor is not None else UniformQE(1.0),
        gap=gap,
    )


# Input-shape helpers


def _as_nx3(value: Sequence[Sequence[float]] | Array, name: str) -> Array:
    """Validate ``value`` as an ``(N, 3)`` array, broadcasting a bare ``(3,)``.

    Raises:
        ValueError: if not 1-D of length 3 nor 2-D with 3 columns.
    """
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) or (3,), got {tuple(arr.shape)}")
    return arr


# SensorGroup ABC


class SensorGroup(eqx.Module):
    """Abstract base class for sensor groups.

    A sensor group contains N sensors at different positions/orientations
    that share the same pixel geometry **and the same detection chain**.
    Each group owns its :class:`~iactrace.camera.chain.DetectionChain`
    (optional concentrator + ``gap`` + photosensor), so distinct groups in
    one :class:`~iactrace.camera.camera.Camera` can carry different cones or
    photosensors.

    Attributes:
        positions: Sensor positions in 3D space (N, 3)
        rotations: Sensor rotations as Euler angles in degrees (N, 3)
        chain: The per-pixel :class:`~iactrace.camera.chain.DetectionChain`
            applied to every pixel of this group.
    """

    positions: Array
    rotations: Array
    chain: DetectionChain

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
    def pixel_index_and_mask(self, sensor_idx: Array, x: Array, y: Array) -> tuple[Array, Array]:
        """Localize ``(x, y)`` to a flat pixel index plus a validity mask.

        The single source of truth shared by :meth:`accumulate` (image path)
        and :meth:`assign_pixels` / :meth:`Camera.collect` (per-ray path), so
        the two can never disagree. Returns ``(flat_idx, valid)`` where *valid*
        is True only for rays that land in a real pixel **and** outside the
        ``edge_width`` dead-zone. Invalid rays get a clamped, meaningless index.
        """
        raise NotImplementedError

    @abstractmethod
    def accumulate(self, sensor_idx: Array, x: Array, y: Array, values: Array) -> Array:
        """Accumulate values at given positions into pixels across all sensors."""
        raise NotImplementedError

    @abstractmethod
    def assign_pixels(self, sensor_idx: Array, x: Array, y: Array) -> Array:
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

    @abstractmethod
    def to_pixel_frame(self, sensor_rays: RayBundle, sensor_idx: Array) -> RayBundle:
        """Re-express tile-local rays in their assigned pixel's local frame.

        A pure isometry: each ray's origin is shifted to the centre of the
        pixel it hit, and origins + directions are rotated into the
        grid-aligned pixel-local frame, so the single shared detection chain
        can be applied to every pixel at once. ``values`` and ``path_length``
        are passed through untouched and origins keep ``z = 0`` (the entrance
        aperture plane). See :mod:`iactrace.camera.chain` for the frame.
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
        concentrator: Concentrator | None = None,
        photosensor: PhotoSensor | None = None,
        gap: float = 0.0,
    ) -> None:
        """Square-pixel sensor group.

        Args:
            positions: Sensor positions, shape ``(N, 3)`` (or ``(3,)`` for one).
            rotations: Euler angles in degrees, same shape as ``positions``.
            width: Pixel count along x (``> 0``).
            height: Pixel count along y (``> 0``).
            bounds: ``(x_min, x_max, y_min, y_max)`` in the sensor-local frame.
            edge_width: Dead-zone width at pixel edges (``>= 0``).
            concentrator: Optional per-pixel light concentrator (e.g. a
                :class:`~iactrace.camera.winston_cone.WinstonCone`).
            photosensor: Per-pixel detector response. ``None`` defaults to a
                perfect flat :class:`~iactrace.camera.photosensor.UniformQE`.
            gap: Spacing from the concentrator exit (or the entrance plane when
                there is no concentrator) to the detector (``>= 0``).

        Raises:
            ValueError: on malformed shapes, non-positive ``width``/``height``,
                degenerate ``bounds``, or negative ``edge_width``.
        """
        self.positions = _as_nx3(positions, "positions")
        self.rotations = _as_nx3(rotations, "rotations")
        if self.positions.shape[0] != self.rotations.shape[0]:
            raise ValueError(
                "positions and rotations must have the same N, got "
                f"{self.positions.shape[0]} and {self.rotations.shape[0]}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}, {height}")
        if edge_width < 0:
            raise ValueError(f"edge_width must be >= 0, got {edge_width}")
        xmin, xmax, ymin, ymax = bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(f"bounds must satisfy x_min < x_max and y_min < y_max, got {bounds}")

        self.width = int(width)
        self.height = int(height)
        self.edge_width = float(edge_width)

        self.x0 = float(xmin)
        self.y0 = float(ymin)
        self.dx = float((xmax - xmin) / width)
        self.dy = float((ymax - ymin) / height)

        self.chain = _build_chain(concentrator, photosensor, gap)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) pixel bounds."""
        return (self.x0, self.x0 + self.dx * self.width, self.y0, self.y0 + self.dy * self.height)

    def get_accumulator_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def pixel_index_and_mask(self, sensor_idx: Array, x: Array, y: Array) -> tuple[Array, Array]:
        x_cont = (x - self.x0) / self.dx
        y_cont = (y - self.y0) / self.dy

        xi = jnp.floor(x_cont).astype(jnp.int32)
        yi = jnp.floor(y_cont).astype(jnp.int32)

        valid = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
        dist_to_edge = _square_edge_distance(x_cont - xi, y_cont - yi, self.dx, self.dy)
        valid = valid & (dist_to_edge >= self.edge_width)

        xi = jnp.clip(xi, 0, self.width - 1)
        yi = jnp.clip(yi, 0, self.height - 1)
        flat_idx = sensor_idx * (self.height * self.width) + yi * self.width + xi
        return flat_idx, valid

    def accumulate(self, sensor_idx: Array, x: Array, y: Array, values: Array) -> Array:
        flat_idx, valid = self.pixel_index_and_mask(sensor_idx, x, y)
        img_flat = jax.ops.segment_sum(
            jnp.where(valid, values, 0.0),
            flat_idx,
            num_segments=self.n_sensors * self.height * self.width,
        )
        return img_flat.reshape(self.n_sensors, self.height, self.width)

    def assign_pixels(self, sensor_idx: Array, x: Array, y: Array) -> Array:
        return self.pixel_index_and_mask(sensor_idx, x, y)[0]

    def in_bounds(self, x: Array, y: Array) -> Array:
        x_max = self.x0 + self.dx * self.width
        y_max = self.y0 + self.dy * self.height
        return (x >= self.x0) & (x <= x_max) & (y >= self.y0) & (y <= y_max)

    def to_pixel_frame(self, sensor_rays: RayBundle, sensor_idx: Array) -> RayBundle:
        x = sensor_rays.origins[:, 0]
        y = sensor_rays.origins[:, 1]
        xi = jnp.floor((x - self.x0) / self.dx)
        yi = jnp.floor((y - self.y0) / self.dy)
        cx = self.x0 + (xi + 0.5) * self.dx
        cy = self.y0 + (yi + 0.5) * self.dy
        local = jnp.stack([x - cx, y - cy, jnp.zeros_like(x)], axis=-1)
        return RayBundle(
            origins=local,
            directions=sensor_rays.directions,
            values=sensor_rays.values,
            path_length=sensor_rays.path_length,
            n=sensor_rays.n,
        )


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
        concentrator: Concentrator | None = None,
        photosensor: PhotoSensor | None = None,
        gap: float = 0.0,
    ) -> None:
        """Hexagonal-pixel sensor group.

        Args:
            positions: Sensor positions, shape ``(N, 3)`` (or ``(3,)`` for one).
            rotations: Euler angles in degrees, same shape as ``positions``.
            hex_centers: Pixel centres, shape ``(M, 2)``. The grid geometry
                (size, rotation, offset, lookup table) is auto-detected from
                these on construction.
            edge_width: Dead-zone width at pixel edges (``>= 0``).
            concentrator: Optional per-pixel light concentrator (e.g. a
                :class:`~iactrace.camera.winston_cone.WinstonCone`).
            photosensor: Per-pixel detector response. ``None`` defaults to a
                perfect flat :class:`~iactrace.camera.photosensor.UniformQE`.
            gap: Spacing from the concentrator exit (or the entrance plane when
                there is no concentrator) to the detector (``>= 0``).

        Raises:
            ValueError: on malformed shapes, empty ``hex_centers``, or negative
                ``edge_width``.
        """
        self.positions = _as_nx3(positions, "positions")
        self.rotations = _as_nx3(rotations, "rotations")
        if self.positions.shape[0] != self.rotations.shape[0]:
            raise ValueError(
                "positions and rotations must have the same N, got "
                f"{self.positions.shape[0]} and {self.rotations.shape[0]}"
            )

        self.hex_centers = jnp.asarray(hex_centers)
        if self.hex_centers.ndim != 2 or self.hex_centers.shape[1] != 2:
            raise ValueError(
                f"hex_centers must have shape (M, 2), got {tuple(self.hex_centers.shape)}"
            )
        if self.hex_centers.shape[0] == 0:
            raise ValueError("hex_centers must be non-empty")
        if edge_width < 0:
            raise ValueError(f"edge_width must be >= 0, got {edge_width}")

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

        self.chain = _build_chain(concentrator, photosensor, gap)

    def get_accumulator_shape(self) -> tuple[int]:
        return (self.n_pixels,)

    def _to_grid_coords(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Transform world coordinates to grid-aligned coordinates."""
        return _rotate(x - self.grid_offset[0], y - self.grid_offset[1], -self.grid_rotation)

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

    def pixel_index_and_mask(self, sensor_idx: Array, x: Array, y: Array) -> tuple[Array, Array]:
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)

        pixel_idx, valid = self._lookup_pixels(qi.astype(jnp.int32), ri.astype(jnp.int32))

        hex_center_x, hex_center_y = _axial_to_cartesian(qi, ri, self.hex_size)
        hex_dist = _hex_norm(x_grid - hex_center_x, y_grid - hex_center_y, self.hex_inradius)
        edge_threshold = 1.0 - self.edge_width / self.hex_inradius
        valid = valid & (hex_dist <= edge_threshold)

        flat_idx = sensor_idx * self.n_pixels + pixel_idx
        return flat_idx, valid

    def accumulate(self, sensor_idx: Array, x: Array, y: Array, values: Array) -> Array:
        flat_idx, valid = self.pixel_index_and_mask(sensor_idx, x, y)
        result = jax.ops.segment_sum(
            jnp.where(valid, values, 0.0),
            flat_idx,
            num_segments=self.n_sensors * self.n_pixels,
        )
        return result.reshape(self.n_sensors, self.n_pixels)

    def assign_pixels(self, sensor_idx: Array, x: Array, y: Array) -> Array:
        return self.pixel_index_and_mask(sensor_idx, x, y)[0]

    def in_bounds(self, x: Array, y: Array) -> Array:
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)
        _, valid = self._lookup_pixels(qi.astype(jnp.int32), ri.astype(jnp.int32))
        return valid

    def to_pixel_frame(self, sensor_rays: RayBundle, sensor_idx: Array) -> RayBundle:
        x = sensor_rays.origins[:, 0]
        y = sensor_rays.origins[:, 1]
        # Work in the grid-aligned frame: subtract the assigned hex centre
        # (rounded in axial space) so the offset is already grid-aligned.
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)
        cx, cy = _axial_to_cartesian(qi, ri, self.hex_size)
        local_x = x_grid - cx
        local_y = y_grid - cy
        # Directions must match the grid-aligned origin frame: rotate the
        # in-plane components by -grid_rotation, leave dz alone.
        dx_g, dy_g = _rotate(
            sensor_rays.directions[:, 0],
            sensor_rays.directions[:, 1],
            -self.grid_rotation,
        )
        return RayBundle(
            origins=jnp.stack([local_x, local_y, jnp.zeros_like(local_x)], axis=-1),
            directions=jnp.stack([dx_g, dy_g, sensor_rays.directions[:, 2]], axis=-1),
            values=sensor_rays.values,
            path_length=sensor_rays.path_length,
            n=sensor_rays.n,
        )
