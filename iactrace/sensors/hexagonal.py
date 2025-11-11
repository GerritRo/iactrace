"""Hexagonal pixel sensors for IACT cameras."""

import jax
import jax.numpy as jnp


def point_to_axial(x, y, hex_size):
    """Convert Cartesian coordinates to fractional axial coordinates."""
    q = (2/3 * x) / hex_size
    r = (-1/3 * x + jnp.sqrt(3)/3 * y) / hex_size
    return q, r


def axial_round(q, r):
    """Round fractional axial coordinates to nearest hex."""
    xgrid = jnp.round(q)
    ygrid = jnp.round(r)
    x = q - xgrid
    y = r - ygrid

    q_out = jnp.where(
        jnp.abs(x) >= jnp.abs(y),
        xgrid + jnp.round(x + 0.5*y),
        xgrid
    )
    r_out = jnp.where(
        jnp.abs(x) >= jnp.abs(y),
        ygrid,
        ygrid + jnp.round(y + 0.5*x)
    )

    return q_out, r_out


def make_accumulate_hex(hex_centers):
    """
    Create a function to accumulate values into hexagonal pixels.

    Args:
        hex_centers: Hexagon center positions (N, 2)

    Returns:
        Accumulation function: (x, y, values) -> image
    """
    # Calculate hex_size from neighbor distances
    diff = hex_centers[:, None, :] - hex_centers[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    dists = jnp.where(jnp.eye(len(hex_centers), dtype=bool), jnp.inf, dists)
    min_dist = jnp.min(dists)
    hex_size = min_dist / jnp.sqrt(3)

    # Get values in axial coordinates
    x = hex_centers[:, 0]
    y = hex_centers[:, 1]
    q, r = point_to_axial(x, y, hex_size)

    # Calculate offset so grid assignments work
    q_offset = q[0] - jnp.round(q[0])
    r_offset = r[0] - jnp.round(r[0])

    q_grid = jnp.round(q - q_offset).astype(jnp.int32)
    r_grid = jnp.round(r - r_offset).astype(jnp.int32)

    # Create 2D lookup table
    q_min, q_max = q_grid.min(), q_grid.max()
    r_min, r_max = r_grid.min(), r_grid.max()

    lookup = jnp.full((q_max - q_min + 1, r_max - r_min + 1), -1, dtype=jnp.int32)
    lookup = lookup.at[q_grid - q_min, r_grid - r_min].set(jnp.arange(len(hex_centers)))

    N = len(hex_centers)

    def accumulate_hex(x, y, V):
        q, r = point_to_axial(x, y, hex_size)
        q_shifted = q - q_offset
        r_shifted = r - r_offset
        q_grid, r_grid = axial_round(q_shifted, r_shifted)

        # Calculate indices
        q_idx = q_grid.astype(jnp.int32) - q_min
        r_idx = r_grid.astype(jnp.int32) - r_min

        # Check bounds BEFORE lookup
        in_bounds = (q_idx >= 0) & (q_idx < lookup.shape[0]) & (r_idx >= 0) & (r_idx < lookup.shape[1])

        # Safe lookup
        q_idx_safe = q_idx.clip(0, lookup.shape[0] - 1)
        r_idx_safe = r_idx.clip(0, lookup.shape[1] - 1)
        idx = lookup[q_idx_safe, r_idx_safe]

        # Valid only if in bounds AND found in lookup
        valid = in_bounds & (idx >= 0)

        idx = jnp.where(valid, idx, 0)
        V_masked = jnp.where(valid, V, 0.0)

        return jax.ops.segment_sum(V_masked, idx, num_segments=N)

    return accumulate_hex


class HexagonalSensor:
    """
    Hexagonal pixel sensor for IACT cameras.

    Currently only supports analytic mode (hard pixel assignment).
    """

    def __init__(self, hex_centers):
        """
        Args:
            hex_centers: Array of hexagon center positions (N, 2)
        """
        self.hex_centers = jnp.array(hex_centers)

    def to_config(self):
        """Convert to configuration dict for compilation."""
        return {
            'type': 'hexagonal',
            'hex_centers': self.hex_centers,
            'mode': 'analytic'
        }

    def make_accumulate_fn(self):
        """Create the accumulation function."""
        return make_accumulate_hex(self.hex_centers)
