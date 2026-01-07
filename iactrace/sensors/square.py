import jax
import jax.numpy as jnp
import equinox as eqx


def _square_edge_distance(x_frac, y_frac, dx, dy):
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


class SquareSensor(eqx.Module):
    """Square pixel sensor."""
    
    position: jax.Array
    rotation: jax.Array
    
    width: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    x0: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)
    
    def __init__(self, position, rotation, width, height, bounds, edge_width=0.0):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.width = int(width)
        self.height = int(height)
        self.edge_width = float(edge_width)
        
        xmin, xmax, ymin, ymax = bounds
        self.x0 = float(xmin)
        self.y0 = float(ymin)
        self.dx = float((xmax - xmin) / width)
        self.dy = float((ymax - ymin) / height)
    
    def get_accumulator_shape(self):
        return (self.height, self.width)
    
    def accumulate(self, x, y, values):
        """Accumulate photon hits into pixels."""
        x_cont = (x - self.x0) / self.dx
        y_cont = (y - self.y0) / self.dy

        xi = jnp.floor(x_cont).astype(jnp.int32)
        yi = jnp.floor(y_cont).astype(jnp.int32)

        valid = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
        
        if self.edge_width > 0:
            x_frac = x_cont - xi
            y_frac = y_cont - yi
            dist_to_edge = _square_edge_distance(x_frac, y_frac, self.dx, self.dy)
            on_edge = dist_to_edge < self.edge_width
            valid = valid & ~on_edge
        
        xi = jnp.clip(xi, 0, self.width - 1)
        yi = jnp.clip(yi, 0, self.height - 1)
        
        values_masked = jnp.where(valid, values, 0.0)
        flat_idx = yi * self.width + xi
        img_flat = jax.ops.segment_sum(
            values_masked, flat_idx, num_segments=self.height * self.width
        )
        
        return img_flat.reshape(self.height, self.width)


class DifferentiableSquareSensor(eqx.Module):
    """Square sensor optimized for differentiable mode."""
    
    position: jax.Array
    rotation: jax.Array
    offset_x: jax.Array
    offset_y: jax.Array
    
    width: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    x0: float = eqx.field(static=True)
    y0: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)
    
    def __init__(self, position, rotation, width, height,
                 bounds=(-1, 1, -1, 1), sigma=0.1, kernel_size=2, edge_width=0.0):
        """Simplified constructor for differentiable mode only."""
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.width = int(width)
        self.height = int(height)
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.edge_width = float(edge_width)
        
        xmin, xmax, ymin, ymax = bounds
        self.x0 = float(xmin)
        self.y0 = float(ymin)
        self.dx = float((xmax - xmin) / width)
        self.dy = float((ymax - ymin) / height)
        
        # Pre-compute offsets
        offset_range = jnp.arange(-kernel_size, kernel_size + 1)
        dx_off, dy_off = jnp.meshgrid(offset_range, offset_range, indexing='xy')
        self.offset_x = dx_off.flatten()
        self.offset_y = dy_off.flatten()
    
    def get_accumulator_shape(self):
        return (self.height, self.width)
    
    def accumulate(self, x, y, values):
        """Differentiable soft splatting."""
        x_pix = (x - self.x0) / self.dx
        y_pix = (y - self.y0) / self.dy
        
        xi_base = jnp.floor(x_pix).astype(jnp.int32)
        yi_base = jnp.floor(y_pix).astype(jnp.int32)
        
        x_frac = x_pix - xi_base
        y_frac = y_pix - yi_base
        
        if self.edge_width > 0:
            dist_to_edge = _square_edge_distance(x_frac, y_frac, self.dx, self.dy)
            on_edge = dist_to_edge < self.edge_width
            # Zero out values for rays hitting edges
            values = jnp.where(on_edge, 0.0, values)
        
        xi = xi_base[:, None] + self.offset_x[None, :]
        yi = yi_base[:, None] + self.offset_y[None, :]
        
        dist_x = x_frac[:, None] - self.offset_x[None, :]
        dist_y = y_frac[:, None] - self.offset_y[None, :]
        weights = jnp.exp(-0.5 * (dist_x**2 + dist_y**2) / self.sigma**2)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)
        
        valid = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
        xi = jnp.clip(xi, 0, self.width - 1)
        yi = jnp.clip(yi, 0, self.height - 1)
        splatted_values = values[:, None] * weights * valid
        
        flat_idx = (yi * self.width + xi).flatten()
        img_flat = jax.ops.segment_sum(
            splatted_values.flatten(), flat_idx, num_segments=self.height * self.width
        )
        
        return img_flat.reshape(self.height, self.width)