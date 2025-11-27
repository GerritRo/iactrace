import jax
import jax.numpy as jnp
import equinox as eqx


class HexagonalSensor(eqx.Module):
    """Hexagonal pixel sensor using axial coordinate lookup."""
    
    position: jax.Array
    rotation: jax.Array
    hex_centers: jax.Array
    lookup_table: jax.Array
    
    hex_size: float = eqx.field(static=True)
    rotation_angle: float = eqx.field(static=True)
    hex_offset_x: float = eqx.field(static=True)
    hex_offset_y: float = eqx.field(static=True)
    q_min: int = eqx.field(static=True)
    r_min: int = eqx.field(static=True)
    n_pixels: int = eqx.field(static=True)
    
    def __init__(self, position, rotation, hex_centers):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        
        # Detect hex properties
        hex_size, rot_angle, offset = _detect_hex_properties(hex_centers)
        self.hex_size = float(hex_size)
        self.rotation_angle = float(rot_angle)
        self.hex_offset_x = float(offset[0])
        self.hex_offset_y = float(offset[1])
        
        # Transform to pointy-top centered grid
        x = hex_centers[:, 0] - self.hex_offset_x
        y = hex_centers[:, 1] - self.hex_offset_y
        x_rot, y_rot = _rotate_coords(x, y, -self.rotation_angle)
        
        q, r = _point_to_axial(x_rot, y_rot, self.hex_size)
        q_grid = jnp.round(q).astype(jnp.int32)
        r_grid = jnp.round(r).astype(jnp.int32)
        
        self.q_min = int(q_grid.min())
        self.r_min = int(r_grid.min())
        q_max = int(q_grid.max())
        r_max = int(r_grid.max())
        
        lookup = jnp.full((q_max - self.q_min + 1, r_max - self.r_min + 1), -1, dtype=jnp.int32)
        lookup = lookup.at[q_grid - self.q_min, r_grid - self.r_min].set(jnp.arange(len(hex_centers)))
        self.lookup_table = lookup
    
    def get_accumulator_shape(self):
        return (self.n_pixels,)
    
    def accumulate(self, x, y, values):
        """Accumulate photon hits into hexagonal pixels."""
        x_rot, y_rot = _rotate_coords(x - self.hex_offset_x, y - self.hex_offset_y, -self.rotation_angle)
        q, r = _point_to_axial(x_rot, y_rot, self.hex_size)
        q_grid, r_grid = _axial_round(q, r)
        
        q_idx = q_grid.astype(jnp.int32) - self.q_min
        r_idx = r_grid.astype(jnp.int32) - self.r_min
        
        in_bounds = (
            (q_idx >= 0) & (q_idx < self.lookup_table.shape[0]) &
            (r_idx >= 0) & (r_idx < self.lookup_table.shape[1])
        )
        
        q_idx_safe = q_idx.clip(0, self.lookup_table.shape[0] - 1)
        r_idx_safe = r_idx.clip(0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_idx_safe, r_idx_safe]
        
        valid = in_bounds & (pixel_idx >= 0)
        pixel_idx = jnp.where(valid, pixel_idx, 0)
        values_masked = jnp.where(valid, values, 0.0)
        
        return jax.ops.segment_sum(values_masked, pixel_idx, num_segments=self.n_pixels)


def _detect_hex_properties(centers):
    """Detect hex size, rotation, and offset from center positions."""
    centers = jnp.asarray(centers)
    N = centers.shape[0]
    
    diff = centers[:, None, :] - centers[None, :, :]
    sqd = jnp.sum(diff**2, axis=2)
    sqd = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, sqd)
    
    flat_index = jnp.argmin(sqd)
    i, j = flat_index // N, flat_index % N
    vec = diff[i, j]
    angle = jnp.arctan2(vec[1], vec[0])
    min_dist = jnp.sqrt(jnp.min(sqd))
    
    offset_id = jnp.argmin(jnp.sqrt(jnp.sum(centers**2, axis=1)))
    return min_dist / jnp.sqrt(3), jnp.mod(angle, jnp.pi/3), centers[offset_id]


def _rotate_coords(x, y, angle):
    """Rotate coordinates by angle."""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def _point_to_axial(x, y, hex_size):
    """Convert Cartesian to axial hex coordinates (pointy-top)."""
    q = (jnp.sqrt(3)/3 * x - 1/3 * y) / hex_size
    r = (2/3 * y) / hex_size
    return q, r


def _axial_round(q, r):
    """Round axial coordinates to nearest hex."""
    s = -q - r
    q_round, r_round, s_round = jnp.round(q), jnp.round(r), jnp.round(s)
    q_diff = jnp.abs(q_round - q)
    r_diff = jnp.abs(r_round - r)
    s_diff = jnp.abs(s_round - s)
    
    q_round = jnp.where((q_diff > r_diff) & (q_diff > s_diff), -r_round - s_round, q_round)
    r_round = jnp.where((r_diff > q_diff) & (r_diff > s_diff), -q_round - s_round, r_round)
    return q_round, r_round
