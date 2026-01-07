"""Hexagonal sensor implementations with hard and differentiable accumulation."""

import jax
import jax.numpy as jnp
import equinox as eqx

# Constants for hexagonal geometry
SQRT3 = 1.7320508075688772
SQRT3_2 = 0.8660254037844386  # sqrt(3)/2
SQRT3_3 = 0.5773502691896257  # sqrt(3)/3 = 1/sqrt(3)


def _rotate(x, y, angle):
    """Rotate 2D coordinates by angle."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return c * x - s * y, s * x + c * y


def _cartesian_to_axial(x, y, size):
    """Cartesian to axial hex coordinates (pointy-top)."""
    return (SQRT3_3 * x - y / 3) / size, (2 * y / 3) / size


def _axial_to_cartesian(q, r, size):
    """Axial to Cartesian hex coordinates (pointy-top)."""
    return size * SQRT3 * (q + r / 2), size * 1.5 * r


def _axial_round(q, r):
    """Round fractional axial coordinates to nearest hex center."""
    s = -q - r
    qi, ri, si = jnp.round(q), jnp.round(r), jnp.round(s)
    dq, dr, ds = jnp.abs(qi - q), jnp.abs(ri - r), jnp.abs(si - s)
    qi = jnp.where((dq > dr) & (dq > ds), -ri - si, qi)
    ri = jnp.where((dr > dq) & (dr > ds), -qi - si, ri)
    return qi, ri


def _hex_norm(x, y, inradius):
    """Hexagonal norm: 0 at center, 1 at boundary (pointy-top).

    The "infinity norm" for hexagonal geometry.
    """
    return jnp.maximum(jnp.abs(y), SQRT3_2 * jnp.abs(x) + 0.5 * jnp.abs(y)) / inradius


def _hex_neighbor_offsets(rings):
    """Generate axial offsets for all hexagons within `rings` distance."""
    offsets = [(q, r) for q in range(-rings, rings + 1)
                      for r in range(-rings, rings + 1)
                      if max(abs(q), abs(r), abs(-q - r)) <= rings]
    return jnp.array([o[0] for o in offsets]), jnp.array([o[1] for o in offsets])


def _detect_hex_grid(centers):
    """Detect hex size, rotation, and offset from center positions."""
    centers = jnp.asarray(centers)
    n = len(centers)

    # Find nearest neighbor distance
    diff = centers[:, None] - centers[None, :]
    dist_sq = jnp.sum(diff ** 2, axis=2)
    dist_sq = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist_sq)
    min_dist = jnp.sqrt(jnp.min(dist_sq))

    # Find rotation from nearest neighbor vector
    idx = jnp.argmin(dist_sq)
    vec = diff[idx // n, idx % n]
    angle = jnp.mod(jnp.arctan2(vec[1], vec[0]), jnp.pi / 3)

    # Find offset (hex center closest to origin)
    offset = centers[jnp.argmin(jnp.sum(centers ** 2, axis=1))]

    return min_dist / SQRT3, angle, offset


def _build_lookup_table(centers, hex_size, rotation, offset):
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
    table = table.at[qi - q_min, ri - r_min].set(jnp.arange(len(centers)))

    return table, q_min, r_min



class HexagonalSensor(eqx.Module):
    """Hexagonal pixel sensor with hard (non-differentiable) accumulation."""

    position: jax.Array
    rotation: jax.Array
    hex_centers: jax.Array
    lookup_table: jax.Array

    hex_size: float = eqx.field(static=True)
    hex_inradius: float = eqx.field(static=True)
    grid_rotation: float = eqx.field(static=True)
    grid_offset: tuple = eqx.field(static=True)
    q_min: int = eqx.field(static=True)
    r_min: int = eqx.field(static=True)
    n_pixels: int = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)

    def __init__(self, position, rotation, hex_centers, edge_width=0.0):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        self.edge_width = float(edge_width)

        # Detect grid properties
        size, rot, offset = _detect_hex_grid(hex_centers)
        self.hex_size = float(size)
        self.hex_inradius = float(size * SQRT3_2)
        self.grid_rotation = float(rot)
        self.grid_offset = (float(offset[0]), float(offset[1]))

        # Build lookup table
        self.lookup_table, self.q_min, self.r_min = _build_lookup_table(
            hex_centers, self.hex_size, self.grid_rotation, offset
        )

    def get_accumulator_shape(self):
        return (self.n_pixels,)

    def _to_grid_coords(self, x, y):
        """Transform world coordinates to grid-aligned coordinates."""
        return _rotate(x - self.grid_offset[0], y - self.grid_offset[1], -self.grid_rotation)

    def _lookup_pixels(self, qi, ri):
        """Look up pixel indices from axial coordinates, handling bounds."""
        q_idx = qi - self.q_min
        r_idx = ri - self.r_min

        in_bounds = ((q_idx >= 0) & (q_idx < self.lookup_table.shape[0]) &
                     (r_idx >= 0) & (r_idx < self.lookup_table.shape[1]))

        q_safe = jnp.clip(q_idx, 0, self.lookup_table.shape[0] - 1)
        r_safe = jnp.clip(r_idx, 0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_safe, r_safe]

        valid = in_bounds & (pixel_idx >= 0)
        return jnp.where(valid, pixel_idx, 0), valid

    def accumulate(self, x, y, values):
        """Accumulate values into hexagonal pixels (hard assignment)."""
        x_grid, y_grid = self._to_grid_coords(x, y)
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        qi, ri = _axial_round(q, r)

        pixel_idx, valid = self._lookup_pixels(qi.astype(jnp.int32), ri.astype(jnp.int32))
        
        if self.edge_width > 0:
            hex_center_x, hex_center_y = _axial_to_cartesian(qi, ri, self.hex_size)
            hex_dist = _hex_norm(x_grid - hex_center_x, y_grid - hex_center_y, self.hex_inradius)
            edge_threshold = 1.0 - self.edge_width / self.hex_inradius
            on_edge = hex_dist > edge_threshold
            valid = valid & ~on_edge
        
        
        return jax.ops.segment_sum(
            jnp.where(valid, values, 0.0), pixel_idx, num_segments=self.n_pixels
        )


class DifferentiableHexagonalSensor(eqx.Module):
    """Hexagonal sensor with differentiable Gaussian splatting."""

    position: jax.Array
    rotation: jax.Array
    hex_centers: jax.Array
    lookup_table: jax.Array
    neighbor_offsets_q: jax.Array
    neighbor_offsets_r: jax.Array
    neighbor_offsets_xy: jax.Array  # Pre-computed Cartesian offsets

    hex_size: float = eqx.field(static=True)
    hex_inradius: float = eqx.field(static=True)
    grid_rotation: float = eqx.field(static=True)
    grid_offset: tuple = eqx.field(static=True)
    q_min: int = eqx.field(static=True)
    r_min: int = eqx.field(static=True)
    n_pixels: int = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)

    def __init__(self, position, rotation, hex_centers, sigma=0.5, edge_width=0.0):
        """Initialize differentiable hexagonal sensor.

        Args:
            position: Sensor position in 3D space
            rotation: Sensor rotation quaternion
            hex_centers: Array of hexagon center positions (N, 2)
            sigma: Gaussian width in units of hex inradius (1.0 = boundary at 1 std dev)
            kernel_size: Number of neighbor rings for splatting
            edge_width: Dead zone width at pixel edges in physical units (default 0.0)
        """
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        self.sigma = float(sigma)
        self.edge_width = float(edge_width)

        # Detect grid properties
        size, rot, offset = _detect_hex_grid(hex_centers)
        self.hex_size = float(size)
        self.hex_inradius = float(size * SQRT3_2)
        self.grid_rotation = float(rot)
        self.grid_offset = (float(offset[0]), float(offset[1]))

        # Build lookup table
        self.lookup_table, self.q_min, self.r_min = _build_lookup_table(
            hex_centers, self.hex_size, self.grid_rotation, offset
        )

        # Pre-compute neighbor offsets (axial and Cartesian)
        self.neighbor_offsets_q, self.neighbor_offsets_r = _hex_neighbor_offsets(kernel_size)
        ox, oy = _axial_to_cartesian(self.neighbor_offsets_q, self.neighbor_offsets_r, self.hex_size)
        self.neighbor_offsets_xy = jnp.stack([ox, oy], axis=1)

    def get_accumulator_shape(self):
        return (self.n_pixels,)

    def accumulate(self, x, y, values):
        """Accumulate values with differentiable Gaussian splatting."""
        # Transform to grid coordinates
        x_grid, y_grid = _rotate(
            x - self.grid_offset[0], y - self.grid_offset[1], -self.grid_rotation
        )

        # Find base hex for each point
        q, r = _cartesian_to_axial(x_grid, y_grid, self.hex_size)
        q_base, r_base = _axial_round(q, r)
        base_x, base_y = _axial_to_cartesian(q_base, r_base, self.hex_size)

        # Position relative to base hex center
        dx = x_grid - base_x
        dy = y_grid - base_y
        
        # Check for edge hits if edge_width > 0 (using distance to primary pixel)
        if self.edge_width > 0:
            base_hex_dist = _hex_norm(dx, dy, self.hex_inradius)
            edge_threshold = 1.0 - self.edge_width / self.hex_inradius
            on_edge = base_hex_dist > edge_threshold
            # Zero out values for rays hitting edges
            values = jnp.where(on_edge, 0.0, values)

        # Broadcast to all neighbors: (n_points, n_neighbors)
        qi = q_base[:, None].astype(jnp.int32) + self.neighbor_offsets_q
        ri = r_base[:, None].astype(jnp.int32) + self.neighbor_offsets_r

        # Distance to each neighbor hex center
        dist_x = dx[:, None] - self.neighbor_offsets_xy[:, 0]
        dist_y = dy[:, None] - self.neighbor_offsets_xy[:, 1]

        # Gaussian weights based on hexagonal distance
        hex_dist = _hex_norm(dist_x, dist_y, self.hex_inradius)
        weights = jnp.exp(-0.5 * (hex_dist / self.sigma) ** 2)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Look up pixel indices
        q_idx = qi - self.q_min
        r_idx = ri - self.r_min

        in_bounds = ((q_idx >= 0) & (q_idx < self.lookup_table.shape[0]) &
                     (r_idx >= 0) & (r_idx < self.lookup_table.shape[1]))

        q_safe = jnp.clip(q_idx, 0, self.lookup_table.shape[0] - 1)
        r_safe = jnp.clip(r_idx, 0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_safe, r_safe]

        valid = in_bounds & (pixel_idx >= 0)
        pixel_idx = jnp.where(valid, pixel_idx, 0)

        # Splat weighted values
        splatted = values[:, None] * weights * valid
        return jax.ops.segment_sum(splatted.flatten(), pixel_idx.flatten(), num_segments=self.n_pixels)