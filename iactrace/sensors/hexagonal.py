"""Hexagonal sensor with efficient axial coordinate lookup."""

import jax
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
    q_offset: float = eqx.field(static=True)
    r_offset: float = eqx.field(static=True)
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
        
        # Compute hex size from neighbor distances
        diff = hex_centers[:, None, :] - hex_centers[None, :, :]
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dists = jnp.where(jnp.eye(len(hex_centers), dtype=bool), jnp.inf, dists)
        min_dist = jnp.min(dists)
        self.hex_size = float(min_dist / jnp.sqrt(3))
        
        # Get axial coordinates
        x, y = hex_centers[:, 0], hex_centers[:, 1]
        q, r = point_to_axial(x, y, self.hex_size)
        
        # Calculate offsets
        self.q_offset = float(q[0] - jnp.round(q[0]))
        self.r_offset = float(r[0] - jnp.round(r[0]))
        
        # Grid coordinates
        q_grid = jnp.round(q - self.q_offset).astype(jnp.int32)
        r_grid = jnp.round(r - self.r_offset).astype(jnp.int32)
        
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
        # Convert to axial coordinates
        q, r = point_to_axial(x, y, self.hex_size)
        
        # Apply offsets
        q_shifted = q - self.q_offset
        r_shifted = r - self.r_offset
        
        # Round to nearest hex
        q_grid, r_grid = axial_round(q_shifted, r_shifted)
        
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


# Helper functions
def point_to_axial(x, y, hex_size):
    """Convert Cartesian to axial hex coordinates."""
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