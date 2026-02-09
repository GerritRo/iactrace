from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .base import SensorGroup

# Constants for hexagonal geometry
SQRT3: float = 1.7320508075688772
SQRT3_2: float = 0.8660254037844386  # sqrt(3)/2
SQRT3_3: float = 0.5773502691896257  # sqrt(3)/3 = 1/sqrt(3)


def _find_three_nearest_hexes_and_weights(
    q: Array, r: Array
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Find 3 nearest hex centers and barycentric weights in axial coordinates.

    For a point at fractional axial coordinates (q, r), finds the 3 nearest
    hex centers that form a triangle containing the point, and computes
    barycentric weights for interpolation.

    Args:
        q: Fractional axial q coordinate
        r: Fractional axial r coordinate

    Returns:
        q0, r0, w0: Center hex (rounded) and its weight
        q1, r1, w1: First neighbor hex and its weight
        q2, r2, w2: Second neighbor hex and its weight
    """
    # Cube coordinate s (constraint: q + r + s = 0)
    s = -q - r

    # Round all three cube coordinates
    qi = jnp.round(q)
    ri = jnp.round(r)
    si = jnp.round(s)

    # Fractional parts (these determine which triangular sector we're in)
    dq = q - qi
    dr = r - ri
    ds = s - si

    # Fix rounding to satisfy q + r + s = 0: adjust the coord with largest error
    # (This is the same logic as _axial_round)
    abs_dq = jnp.abs(dq)
    abs_dr = jnp.abs(dr)
    abs_ds = jnp.abs(ds)

    q0 = jnp.where((abs_dq > abs_dr) & (abs_dq > abs_ds), -ri - si, qi)
    r0 = jnp.where((abs_dr > abs_dq) & (abs_dr > abs_ds), -q0 - si, ri)

    # Recalculate fractional parts relative to corrected center
    s0 = -q0 - r0
    dq = q - q0
    dr = r - r0
    ds = s - s0

    # Pattern: max |d| determines which axis is "crossed", other two signs give direction
    max_is_q = (abs_dq >= abs_dr) & (abs_dq >= abs_ds)
    max_is_r = (abs_dr > abs_dq) & (abs_dr >= abs_ds)

    # Neighbor 1: the neighbor in the direction of the largest |d|
    dq1 = jnp.where(max_is_q, jnp.sign(dq), jnp.where(max_is_r, 0.0, -jnp.sign(ds)))
    dr1 = jnp.where(max_is_q, 0.0, jnp.where(max_is_r, jnp.sign(dr), jnp.sign(ds)))

    # Neighbor 2: determined by the secondary direction
    dq2 = jnp.where(
        max_is_q,
        jnp.sign(dq),
        jnp.where(max_is_r, -jnp.sign(dr), jnp.where(dq >= 0, 1.0, -1.0)),
    )
    dr2 = jnp.where(
        max_is_q,
        jnp.where(dr >= 0, 1.0, -1.0),
        jnp.where(max_is_r, jnp.sign(dr), 0.0),
    )

    # Ensure cube constraint for neighbors: dq + dr + ds = 0 for offset

    q1 = q0 + dq1
    r1 = r0 + dr1
    q2 = q0 + dq2
    r2 = r0 + dr2

    # Compute barycentric weights using the fractional positions
    px = SQRT3 * (dq + dr / 2)
    py = 1.5 * dr

    # Neighbor 1 cartesian offset
    n1x = SQRT3 * (dq1 + dr1 / 2)
    n1y = 1.5 * dr1

    # Neighbor 2 cartesian offset
    n2x = SQRT3 * (dq2 + dr2 / 2)
    n2y = 1.5 * dr2

    denom = n2y * n1x - n2x * n1y
    # Avoid division by zero (shouldn't happen for valid hex triangles)
    denom = jnp.where(jnp.abs(denom) < 1e-10, 1e-10, denom)

    w1 = (n2y * px - n2x * py) / denom
    w2 = (-n1y * px + n1x * py) / denom
    w0 = 1.0 - w1 - w2

    # Clamp weights to [0, 1] for numerical stability at boundaries
    w0 = jnp.clip(w0, 0.0, 1.0)
    w1 = jnp.clip(w1, 0.0, 1.0)
    w2 = jnp.clip(w2, 0.0, 1.0)

    # Renormalize
    w_sum = w0 + w1 + w2
    w_sum = jnp.where(w_sum < 1e-10, 1.0, w_sum)
    w0 = w0 / w_sum
    w1 = w1 / w_sum
    w2 = w2 / w_sum

    return (
        q0.astype(jnp.int32),
        r0.astype(jnp.int32),
        w0,
        q1.astype(jnp.int32),
        r1.astype(jnp.int32),
        w1,
        q2.astype(jnp.int32),
        r2.astype(jnp.int32),
        w2,
    )


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
    """Hexagonal norm: 0 at center, 1 at boundary (pointy-top).

    The "infinity norm" for hexagonal geometry.
    """
    return jnp.maximum(jnp.abs(x), 0.5 * jnp.abs(x) + SQRT3_2 * jnp.abs(y)) / inradius


def _hex_neighbor_offsets(rings: int) -> tuple[Array, Array]:
    """Generate axial offsets for all hexagons within `rings` distance."""
    offsets = [
        (q, r)
        for q in range(-rings, rings + 1)
        for r in range(-rings, rings + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= rings
    ]
    return jnp.array([o[0] for o in offsets]), jnp.array([o[1] for o in offsets])


def _detect_hex_grid(centers: Array) -> tuple[Array, Array, Array]:
    """Detect hex size, rotation, and offset from center positions."""
    centers = jnp.asarray(centers)
    n = len(centers)

    # Find nearest neighbor distance
    diff = centers[:, None] - centers[None, :]
    dist_sq = jnp.sum(diff**2, axis=2)
    dist_sq = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist_sq)
    min_dist = jnp.sqrt(jnp.min(dist_sq))

    # Find rotation from nearest neighbor vector
    idx = jnp.argmin(dist_sq)
    vec = diff[idx // n, idx % n]
    angle = jnp.mod(jnp.arctan2(vec[1], vec[0]), jnp.pi / 3)

    # Find offset (hex center closest to origin)
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
    # Use int32 for indices and values to avoid type promotion warnings
    table = table.at[qi - jnp.int32(q_min), ri - jnp.int32(r_min)].set(
        jnp.arange(len(centers), dtype=jnp.int32)
    )

    return table, q_min, r_min


class HexagonalSensorGroup(SensorGroup):
    """Hexagonal pixel sensor group with hard (non-differentiable) accumulation.

    Contains N sensors at different positions/orientations that share the same
    hexagonal pixel geometry. Output shape is (N, n_pixels).
    """

    config_type: ClassVar[str] = "hexagonal"

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
        """Initialize hexagonal sensor group.

        Args:
            positions: Sensor positions (N, 3) in world coordinates
            rotations: Sensor rotations (N, 3) as Euler angles in degrees
            hex_centers: Hexagon center positions (M, 2) - shared by all sensors
            edge_width: Width of edge exclusion zone
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

        # Ensure 2D arrays even for single sensor
        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if self.rotations.ndim == 1:
            self.rotations = self.rotations[None, :]

        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        self.edge_width = float(edge_width)

        # Detect grid properties
        size, rot, offset = _detect_hex_grid(self.hex_centers)
        self.hex_size = float(size)
        self.hex_inradius = float(size * SQRT3_2)
        self.grid_rotation = float(rot)
        self.grid_offset = (float(offset[0]), float(offset[1]))

        # Build lookup table
        self.lookup_table, self.q_min, self.r_min = _build_lookup_table(
            self.hex_centers, self.hex_size, self.grid_rotation, offset
        )

    def get_accumulator_shape(self) -> tuple[int]:
        """Return per-sensor accumulator shape: (n_pixels,)."""
        return (self.n_pixels,)

    def to_config(self, index: int = 0) -> dict[str, Any]:
        """Convert sensor at index to config dict."""
        hex_centers = np.asarray(self.hex_centers)
        config: dict[str, Any] = {
            "type": "hexagonal",
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "centers_x": [float(c) for c in hex_centers[:, 0]],
            "centers_y": [float(c) for c in hex_centers[:, 1]],
        }
        if self.edge_width > 0:
            config["edge_width"] = self.edge_width
        return config

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> HexagonalSensorGroup:
        """Create HexagonalSensorGroup from list of config dicts.

        All sensors must share the same hexagonal pixel geometry.
        """
        if not configs:
            raise ValueError("At least one sensor config is required")

        # Use first config to get shared geometry
        first = configs[0]
        centers_x = first["centers_x"]
        centers_y = first["centers_y"]
        hex_centers = [[x, y] for x, y in zip(centers_x, centers_y, strict=False)]
        edge_width = first.get("edge_width", 0.0)

        # Collect positions and rotations
        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]

        return cls(
            positions=positions,
            rotations=rotations,
            hex_centers=hex_centers,
            edge_width=edge_width,
        )

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
        """Accumulate values into hexagonal pixels across all sensors.

        Args:
            sensor_idx: Index of sensor each ray hit (n_rays,)
            x: X coordinates in local sensor planes (n_rays,)
            y: Y coordinates in local sensor planes (n_rays,)
            values: Values to accumulate (n_rays,)

        Returns:
            Accumulated values (n_sensors, n_pixels)
        """
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

        # 2D flat index: sensor_idx * n_pixels + pixel_idx
        flat_idx = sensor_idx * self.n_pixels + pixel_idx

        result = jax.ops.segment_sum(
            jnp.where(valid, values, 0.0), flat_idx,
            num_segments=self.n_sensors * self.n_pixels
        )
        return result.reshape(self.n_sensors, self.n_pixels)


def _hex_ste_forward(
    sensor_idx: Array,
    x: Array,
    y: Array,
    values: Array,
    n_sensors: int,
    grid_offset: tuple[float, float],
    grid_rotation: float,
    hex_size: float,
    hex_inradius: float,
    lookup_table: Array,
    q_min: int,
    r_min: int,
    n_pixels: int,
    edge_width: float,
) -> Array:
    """Forward pass: hard assignment (identical to HexagonalSensorGroup)."""
    x_grid, y_grid = _rotate(
        x - grid_offset[0], y - grid_offset[1], -grid_rotation
    )
    q, r = _cartesian_to_axial(x_grid, y_grid, hex_size)
    qi, ri = _axial_round(q, r)

    # Lookup pixel index (use int32 for q_min/r_min to avoid type promotion)
    q_idx = qi.astype(jnp.int32) - jnp.int32(q_min)
    r_idx = ri.astype(jnp.int32) - jnp.int32(r_min)

    in_bounds = (
        (q_idx >= 0)
        & (q_idx < lookup_table.shape[0])
        & (r_idx >= 0)
        & (r_idx < lookup_table.shape[1])
    )

    q_safe = jnp.clip(q_idx, 0, lookup_table.shape[0] - 1)
    r_safe = jnp.clip(r_idx, 0, lookup_table.shape[1] - 1)
    pixel_idx = lookup_table[q_safe, r_safe]

    valid = in_bounds & (pixel_idx >= 0)

    # Edge exclusion
    hex_center_x, hex_center_y = _axial_to_cartesian(qi, ri, hex_size)
    hex_dist = _hex_norm(x_grid - hex_center_x, y_grid - hex_center_y, hex_inradius)
    edge_threshold = 1.0 - edge_width / hex_inradius
    on_edge = hex_dist > edge_threshold
    valid = valid & ~on_edge

    pixel_idx = jnp.where(valid, pixel_idx, 0)

    # 2D flat index: sensor_idx * n_pixels + pixel_idx
    flat_idx = sensor_idx * n_pixels + pixel_idx

    result = jax.ops.segment_sum(
        jnp.where(valid, values, 0.0), flat_idx, num_segments=n_sensors * n_pixels
    )
    return result.reshape(n_sensors, n_pixels)


def _hex_ste_backward(
    sensor_idx: Array,
    x: Array,
    y: Array,
    values: Array,
    g: Array,
    n_sensors: int,
    grid_offset: tuple[float, float],
    grid_rotation: float,
    hex_size: float,
    hex_inradius: float,
    lookup_table: Array,
    q_min: int,
    r_min: int,
    n_pixels: int,
    edge_width: float,
) -> tuple[Array, Array, Array]:
    """Backward pass: barycentric interpolation gradients to 3 nearest hexes."""
    # Transform to grid coordinates
    x_grid, y_grid = _rotate(
        x - grid_offset[0], y - grid_offset[1], -grid_rotation
    )
    q, r = _cartesian_to_axial(x_grid, y_grid, hex_size)

    # Find 3 nearest hexes and barycentric weights
    q0, r0, w0, q1, r1, w1, q2, r2, w2 = _find_three_nearest_hexes_and_weights(q, r)

    # Convert q_min/r_min to int32 to avoid type promotion warnings
    q_min_i32 = jnp.int32(q_min)
    r_min_i32 = jnp.int32(r_min)

    def lookup_pixel_and_valid(qi: Array, ri: Array) -> tuple[Array, Array]:
        """Helper to look up pixel index and validity."""
        q_idx = qi - q_min_i32
        r_idx = ri - r_min_i32
        in_bounds = (
            (q_idx >= 0)
            & (q_idx < lookup_table.shape[0])
            & (r_idx >= 0)
            & (r_idx < lookup_table.shape[1])
        )
        q_safe = jnp.clip(q_idx, 0, lookup_table.shape[0] - 1)
        r_safe = jnp.clip(r_idx, 0, lookup_table.shape[1] - 1)
        pixel_idx = lookup_table[q_safe, r_safe]
        valid = in_bounds & (pixel_idx >= 0)
        return jnp.where(valid, pixel_idx, 0), valid

    # Look up all 3 hexes
    idx0, valid0 = lookup_pixel_and_valid(q0, r0)
    idx1, valid1 = lookup_pixel_and_valid(q1, r1)
    idx2, valid2 = lookup_pixel_and_valid(q2, r2)

    # Edge exclusion based on center hex position
    qi_center, ri_center = _axial_round(q, r)
    hex_center_x, hex_center_y = _axial_to_cartesian(qi_center, ri_center, hex_size)
    hex_dist = _hex_norm(x_grid - hex_center_x, y_grid - hex_center_y, hex_inradius)
    edge_threshold = 1.0 - edge_width / hex_inradius
    on_edge = hex_dist > edge_threshold

    # Overall validity: all 3 hexes must be valid and not on edge
    valid = valid0 & valid1 & valid2 & ~on_edge

    # Get output gradients at each of the 3 hex pixels
    # Use 2D indexing: g[sensor_idx, pixel_idx]
    g_flat = g.reshape(-1)
    g0 = g_flat[sensor_idx * n_pixels + idx0]
    g1 = g_flat[sensor_idx * n_pixels + idx1]
    g2 = g_flat[sensor_idx * n_pixels + idx2]

    # Gradient w.r.t. values: weighted sum of output gradients
    grad_values = jnp.where(valid, w0 * g0 + w1 * g1 + w2 * g2, 0.0)

    # Compute gradient via finite differences (more robust than analytical for complex hex geometry)
    eps = 1e-5 * hex_size

    # Perturb x_grid
    q_px, r_px = _cartesian_to_axial(x_grid + eps, y_grid, hex_size)
    _, _, w0_px, _, _, w1_px, _, _, w2_px = _find_three_nearest_hexes_and_weights(
        q_px, r_px
    )

    q_mx, r_mx = _cartesian_to_axial(x_grid - eps, y_grid, hex_size)
    _, _, w0_mx, _, _, w1_mx, _, _, w2_mx = _find_three_nearest_hexes_and_weights(
        q_mx, r_mx
    )

    dw0_dx = (w0_px - w0_mx) / (2 * eps)
    dw1_dx = (w1_px - w1_mx) / (2 * eps)
    dw2_dx = (w2_px - w2_mx) / (2 * eps)

    # Perturb y_grid
    q_py, r_py = _cartesian_to_axial(x_grid, y_grid + eps, hex_size)
    _, _, w0_py, _, _, w1_py, _, _, w2_py = _find_three_nearest_hexes_and_weights(
        q_py, r_py
    )

    q_my, r_my = _cartesian_to_axial(x_grid, y_grid - eps, hex_size)
    _, _, w0_my, _, _, w1_my, _, _, w2_my = _find_three_nearest_hexes_and_weights(
        q_my, r_my
    )

    dw0_dy = (w0_py - w0_my) / (2 * eps)
    dw1_dy = (w1_py - w1_my) / (2 * eps)
    dw2_dy = (w2_py - w2_my) / (2 * eps)

    # Chain rule: grad_x_grid = values * sum(d(weight)/d(x_grid) * g_pixel)
    grad_x_grid = values * (dw0_dx * g0 + dw1_dx * g1 + dw2_dx * g2)
    grad_y_grid = values * (dw0_dy * g0 + dw1_dy * g1 + dw2_dy * g2)

    # Transform gradients back to world coordinates
    c = jnp.cos(-grid_rotation)
    s = jnp.sin(-grid_rotation)

    grad_x = jnp.where(valid, grad_x_grid * c + grad_y_grid * s, 0.0)
    grad_y = jnp.where(valid, -grad_x_grid * s + grad_y_grid * c, 0.0)

    return grad_x, grad_y, grad_values


def _make_ste_hex_accumulate(
    n_sensors: int,
    grid_offset: tuple[float, float],
    grid_rotation: float,
    hex_size: float,
    hex_inradius: float,
    q_min: int,
    r_min: int,
    n_pixels: int,
    edge_width: float,
):
    """Create a straight-through estimator accumulate function with custom_vjp.

    Note: lookup_table is passed as an argument rather than closed over to avoid
    issues with JAX tracing (closing over traced arrays causes "No constant handler" errors).
    """

    @jax.custom_vjp
    def ste_accumulate(
        sensor_idx: Array, x: Array, y: Array, values: Array, lookup_table: Array
    ) -> Array:
        return _hex_ste_forward(
            sensor_idx,
            x,
            y,
            values,
            n_sensors,
            grid_offset,
            grid_rotation,
            hex_size,
            hex_inradius,
            lookup_table,
            q_min,
            r_min,
            n_pixels,
            edge_width,
        )

    def ste_accumulate_fwd(
        sensor_idx: Array, x: Array, y: Array, values: Array, lookup_table: Array
    ):
        result = _hex_ste_forward(
            sensor_idx,
            x,
            y,
            values,
            n_sensors,
            grid_offset,
            grid_rotation,
            hex_size,
            hex_inradius,
            lookup_table,
            q_min,
            r_min,
            n_pixels,
            edge_width,
        )
        return result, (sensor_idx, x, y, values, lookup_table)

    def ste_accumulate_bwd(residuals, g):
        sensor_idx, x, y, values, lookup_table = residuals
        grad_x, grad_y, grad_values = _hex_ste_backward(
            sensor_idx,
            x,
            y,
            values,
            g,
            n_sensors,
            grid_offset,
            grid_rotation,
            hex_size,
            hex_inradius,
            lookup_table,
            q_min,
            r_min,
            n_pixels,
            edge_width,
        )
        # Return None gradient for sensor_idx and lookup_table
        return None, grad_x, grad_y, grad_values, None

    ste_accumulate.defvjp(ste_accumulate_fwd, ste_accumulate_bwd)
    return ste_accumulate


class StraightThroughHexagonalSensorGroup(SensorGroup):
    """Hexagonal sensor group with straight-through estimator.

    Forward pass uses hard assignment (like HexagonalSensorGroup).
    Backward pass uses barycentric interpolation to 3 nearest hex centers.

    Note: This is not registered with sensor_registry because YAML configs
    always load as HexagonalSensorGroup. Use telescope.with_ste() to convert.
    """

    # Use same config_type as HexagonalSensorGroup for serialization compatibility
    config_type = "hexagonal"

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
        """Initialize straight-through hexagonal sensor group.

        Args:
            positions: Sensor positions (N, 3) in world coordinates
            rotations: Sensor rotations (N, 3) as Euler angles in degrees
            hex_centers: Hexagon center positions (M, 2) - shared by all sensors
            edge_width: Width of edge exclusion zone
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)

        # Ensure 2D arrays even for single sensor
        if self.positions.ndim == 1:
            self.positions = self.positions[None, :]
        if self.rotations.ndim == 1:
            self.rotations = self.rotations[None, :]

        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        self.edge_width = float(edge_width)

        # Detect grid properties
        size, rot, offset = _detect_hex_grid(self.hex_centers)
        self.hex_size = float(size)
        self.hex_inradius = float(size * SQRT3_2)
        self.grid_rotation = float(rot)
        self.grid_offset = (float(offset[0]), float(offset[1]))

        # Build lookup table
        self.lookup_table, self.q_min, self.r_min = _build_lookup_table(
            self.hex_centers, self.hex_size, self.grid_rotation, offset
        )

    def get_accumulator_shape(self) -> tuple[int]:
        """Return per-sensor accumulator shape: (n_pixels,)."""
        return (self.n_pixels,)

    def to_config(self, index: int = 0) -> dict[str, Any]:
        """Convert sensor at index to config dict."""
        hex_centers = np.asarray(self.hex_centers)
        config: dict[str, Any] = {
            "type": "hexagonal",
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "centers_x": [float(c) for c in hex_centers[:, 0]],
            "centers_y": [float(c) for c in hex_centers[:, 1]],
        }
        if self.edge_width > 0:
            config["edge_width"] = self.edge_width
        return config

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> StraightThroughHexagonalSensorGroup:
        """Create StraightThroughHexagonalSensorGroup from list of config dicts."""
        if not configs:
            raise ValueError("At least one sensor config is required")

        first = configs[0]
        centers_x = first["centers_x"]
        centers_y = first["centers_y"]
        hex_centers = [[x, y] for x, y in zip(centers_x, centers_y, strict=False)]

        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]

        return cls(
            positions=positions,
            rotations=rotations,
            hex_centers=hex_centers,
            edge_width=first.get("edge_width", 0.0),
        )

    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        """Accumulate with straight-through estimator.

        Forward: hard assignment to single hex pixel.
        Backward: barycentric interpolation gradients to 3 nearest hexes.
        """
        ste_fn = _make_ste_hex_accumulate(
            self.n_sensors,
            self.grid_offset,
            self.grid_rotation,
            self.hex_size,
            self.hex_inradius,
            self.q_min,
            self.r_min,
            self.n_pixels,
            self.edge_width,
        )
        return ste_fn(sensor_idx, x, y, values, self.lookup_table)
