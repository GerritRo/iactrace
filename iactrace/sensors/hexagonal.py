import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx


class HexagonalSensor(eqx.Module):
    """Hexagonal sensor using axial coordinate lookup."""
    
    # Sensor pose
    position: jax.Array  # (3,)
    rotation: jax.Array  # (3,)
    
    # Hexagon geometry
    hex_centers: jax.Array  # (N, 2)
    lookup_table: jax.Array  # (Q, R) -> pixel index
    
    # Pre-computed static parameters
    hex_size: float = eqx.field(static=True)
    rotation_angle: float = eqx.field(static=True)  # Rotation to align to pointy-top
    hex_offset_x: float = eqx.field(static=True)
    hex_offset_y: float = eqx.field(static=True)
    
    
    q_min: int = eqx.field(static=True)
    r_min: int = eqx.field(static=True)
    n_pixels: int = eqx.field(static=True)
    
    def __init__(self, position, rotation, hex_centers):
        """
        Initialize hexagonal sensor with axial coordinate lookup.
        
        Args:
            position: Sensor position (3,)
            rotation: Sensor euler angles (3,)
            hex_centers: Hexagon center positions (N, 2)
        """
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.hex_centers = jnp.asarray(hex_centers)
        self.n_pixels = len(hex_centers)
        
        # Detect hex properties
        hex_size, rotation, offset = detect_hex_properties(hex_centers)
        self.hex_size = float(hex_size)
        self.rotation_angle = float(rotation)
        self.hex_offset_x = float(offset[0])
        self.hex_offset_y = float(offset[1])
        
        # Rotate and translate to fit pointy-top centered grid
        x = hex_centers[:, 0] - self.hex_offset_x
        y = hex_centers[:, 1] - self.hex_offset_y 
        x_rot, y_rot = rotate_coords(x, y, -self.rotation_angle)
        
        # Get axial coordinates (always using pointy-top)
        q, r = point_to_axial(x_rot, y_rot, self.hex_size)
                
        # Grid coordinates
        q_grid = jnp.round(q).astype(jnp.int32)
        r_grid = jnp.round(r).astype(jnp.int32)
        
        # Store bounds
        self.q_min = int(q_grid.min())
        self.r_min = int(r_grid.min())
        q_max = int(q_grid.max())
        r_max = int(r_grid.max())
        
        # Create lookup table
        lookup = jnp.full(
            (q_max - self.q_min + 1, r_max - self.r_min + 1),
            -1,
            dtype=jnp.int32
        )
        lookup = lookup.at[q_grid - self.q_min, r_grid - self.r_min].set(
            jnp.arange(len(hex_centers))
        )
        self.lookup_table = lookup
    
    def get_accumulator_shape(self):
        """Return shape of accumulator array."""
        return (self.n_pixels,)
    
    def accumulate(self, x, y, values):
        """
        Accumulate photon hits into hexagonal pixels.
        
        Args:
            x: X coordinates of photons (N,)
            y: Y coordinates of photons (N,)
            values: Values to accumulate (N,)
        
        Returns:
            Image array (n_pixels,)
        """
        # Rotate coordinates to align with pointy-top
        x_rot, y_rot = rotate_coords(x-self.hex_offset_x, y-self.hex_offset_y, -self.rotation_angle)
        
        # Convert to axial coordinates (always using pointy-top)
        q, r = point_to_axial(x_rot, y_rot, self.hex_size)
        
        # Round to nearest hex
        q_grid, r_grid = axial_round(q, r)
        
        # Calculate lookup indices
        q_idx = q_grid.astype(jnp.int32) - self.q_min
        r_idx = r_grid.astype(jnp.int32) - self.r_min
        
        # Check bounds
        in_bounds = (
            (q_idx >= 0) & (q_idx < self.lookup_table.shape[0]) &
            (r_idx >= 0) & (r_idx < self.lookup_table.shape[1])
        )
        
        # Safe lookup (clamp indices)
        q_idx_safe = q_idx.clip(0, self.lookup_table.shape[0] - 1)
        r_idx_safe = r_idx.clip(0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_idx_safe, r_idx_safe]
        
        # Valid only if in bounds AND found in lookup
        valid = in_bounds & (pixel_idx >= 0)
        pixel_idx = jnp.where(valid, pixel_idx, 0)
        values_masked = jnp.where(valid, values, 0.0)
        
        # Accumulate using segment_sum
        return jax.ops.segment_sum(
            values_masked,
            pixel_idx,
            num_segments=self.n_pixels
        )
    
    def accumulate_debug(self, x, y, values):
        """
        Accumulate photon hits into hexagonal pixels.
        
        Args:
            x: X coordinates of photons (N,)
            y: Y coordinates of photons (N,)
            values: Values to accumulate (N,)
        
        Returns:
            Image array (n_pixels,)
        """
        # Rotate coordinates to align with pointy-top
        x_rot, y_rot = rotate_coords(x-self.hex_offset_x, y-self.hex_offset_y, -self.rotation_angle)
        
        # Convert to axial coordinates (always using pointy-top)
        q, r = point_to_axial(x_rot, y_rot, self.hex_size)
        
        # Round to nearest hex
        q_grid, r_grid = axial_round(q, r)
        
        # Calculate lookup indices
        q_idx = q_grid.astype(jnp.int32) - self.q_min
        r_idx = r_grid.astype(jnp.int32) - self.r_min
        
        # Check bounds
        in_bounds = (
            (q_idx >= 0) & (q_idx < self.lookup_table.shape[0]) &
            (r_idx >= 0) & (r_idx < self.lookup_table.shape[1])
        )
        
        # Safe lookup (clamp indices)
        q_idx_safe = q_idx.clip(0, self.lookup_table.shape[0] - 1)
        r_idx_safe = r_idx.clip(0, self.lookup_table.shape[1] - 1)
        pixel_idx = self.lookup_table[q_idx_safe, r_idx_safe]
        
        # Accumulate using segment_sum
        return pixel_idx, in_bounds & (pixel_idx >= 0)


# Helper functions
def detect_hex_properties(centers):
    centers = jnp.asarray(centers)
    N = centers.shape[0]

    # Pairwise differences and distances
    diff = centers[:, None, :] - centers[None, :, :]
    sqd = jnp.sum(diff**2, axis=2)

    # mask diagonal
    diag_mask = jnp.eye(N, dtype=bool)
    sqd = jnp.where(diag_mask, jnp.inf, sqd)

    # Find the minimum
    # argmin over flattened array
    flat_index = jnp.argmin(sqd)
    i = flat_index // N
    j = flat_index % N

    # minimal vector
    vec = diff[i, j]
    
    # angle of this vector
    angle = jnp.arctan2(vec[1], vec[0])
    
    # Minimum distance
    min_dist = jnp.sqrt(jnp.min(sqd))
    
    # Calculate offset via smallest distance to center:
    offset_id = jnp.argmin(jnp.sqrt(jnp.sum(centers**2, axis=1)))

    return min_dist/jnp.sqrt(3), jnp.mod(angle, jnp.pi/3), centers[offset_id]


def rotate_coords(x, y, angle):
    """Rotate coordinates by given angle."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return x_rot, y_rot


def point_to_axial(x, y, hex_size):
    """Convert Cartesian to axial hex coordinates (pointy-top)."""
    q = (jnp.sqrt(3)/3 * x - 1/3 * y) / hex_size
    r = (2/3 * y) / hex_size
    return q, r


def axial_round(q, r):
    """Round axial coordinates to nearest hex."""
    s = -q - r
    
    q_round = jnp.round(q)
    r_round = jnp.round(r)
    s_round = jnp.round(s)
    
    q_diff = jnp.abs(q_round - q)
    r_diff = jnp.abs(r_round - r)
    s_diff = jnp.abs(s_round - s)
    
    # Reset the component with largest rounding error
    q_round = jnp.where(
        (q_diff > r_diff) & (q_diff > s_diff),
        -r_round - s_round,
        q_round
    )
    r_round = jnp.where(
        (r_diff > q_diff) & (r_diff > s_diff),
        -q_round - s_round,
        r_round
    )
    
    return q_round, r_round