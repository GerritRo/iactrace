"""Square pixel sensors with both analytic and differentiable modes."""

import jax
import jax.numpy as jnp


def make_accumulate_square(H, W, x0=0.0, y0=0.0, dx=1.0, dy=1.0):
    """
    Create a function to accumulate values into square pixels (analytic mode).

    Args:
        H: Height in pixels
        W: Width in pixels
        x0, y0: Origin of pixel grid
        dx, dy: Pixel spacing

    Returns:
        Accumulation function: (x, y, values) -> image
    """
    def accumulate_square(x, y, V):
        # Compute pixel indices
        xi = jnp.floor((x - x0) / dx).astype(jnp.int32)
        yi = jnp.floor((y - y0) / dy).astype(jnp.int32)

        # Mask for valid indices
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

        # Clip indices to valid range
        xi = jnp.clip(xi, 0, W - 1)
        yi = jnp.clip(yi, 0, H - 1)

        # Zero out invalid values
        V_masked = jnp.where(valid, V, 0.0)

        # Flatten and accumulate
        flat_idx = yi * W + xi
        img_flat = jax.ops.segment_sum(V_masked, flat_idx, num_segments=H*W)

        return img_flat.reshape(H, W)

    return accumulate_square


def make_accumulate_square_splats(H, W, sigma=0.1, kernel_size=2,
                                   x0=0.0, y0=0.0, dx=1.0, dy=1.0):
    """
    Create a function to accumulate values with soft splatting (differentiable mode).

    Args:
        H: Height in pixels
        W: Width in pixels
        sigma: Gaussian width for splatting
        kernel_size: Radius of splat kernel in pixels
        x0, y0: Origin of pixel grid
        dx, dy: Pixel spacing

    Returns:
        Accumulation function: (x, y, values) -> image
    """
    offset_range = jnp.arange(-kernel_size, kernel_size + 1)
    dx_off, dy_off = jnp.meshgrid(offset_range, offset_range, indexing='xy')
    dx_flat = dx_off.flatten()
    dy_flat = dy_off.flatten()

    def accumulate_square_splats(x, y, V):
        x_pix = (x - x0) / dx
        y_pix = (y - y0) / dy

        # Base pixel (integer)
        xi_base = jnp.floor(x_pix).astype(jnp.int32)
        yi_base = jnp.floor(y_pix).astype(jnp.int32)

        # Sub-pixel offset (continuous, gradients flow through this)
        x_frac = x_pix - xi_base
        y_frac = y_pix - yi_base

        # Neighbor indices
        xi = xi_base[:, None] + dx_flat[None, :]
        yi = yi_base[:, None] + dy_flat[None, :]

        # Compute actual distances to pixel centers
        dist_x = x_frac[:, None] - dx_flat[None, :]
        dist_y = y_frac[:, None] - dy_flat[None, :]
        weights = jnp.exp(-0.5 * (dist_x**2 + dist_y**2) / sigma**2)

        # Normalize weights per point
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Valid mask and accumulate
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi = jnp.clip(xi, 0, W - 1)
        yi = jnp.clip(yi, 0, H - 1)
        values = V[:, None] * weights * valid

        flat_idx = (yi * W + xi).flatten()
        img_flat = jax.ops.segment_sum(values.flatten(), flat_idx, num_segments=H*W)
        return img_flat.reshape(H, W)

    return accumulate_square_splats


class SquareSensor:
    """
    Square pixel sensor for telescope imaging.

    Supports both analytic (fast, non-differentiable) and differentiable modes.
    """

    def __init__(self, width, height, bounds=(-1, 1, -1, 1), mode='analytic',
                 sigma=0.1, kernel_size=2):
        """
        Args:
            width: Number of pixels horizontally
            height: Number of pixels vertically
            bounds: (xmin, xmax, ymin, ymax) in world coordinates
            mode: 'analytic' or 'differentiable'
            sigma: Gaussian width for differentiable mode
            kernel_size: Splat radius for differentiable mode
        """
        self.width = width
        self.height = height
        self.bounds = bounds
        self.mode = mode
        self.sigma = sigma
        self.kernel_size = kernel_size

        xmin, xmax, ymin, ymax = bounds
        self.x0 = xmin
        self.y0 = ymin
        self.dx = (xmax - xmin) / width
        self.dy = (ymax - ymin) / height

    def to_config(self):
        """Convert to configuration dict for compilation."""
        return {
            'type': 'square',
            'width': self.width,
            'height': self.height,
            'x0': self.x0,
            'y0': self.y0,
            'dx': self.dx,
            'dy': self.dy,
            'mode': self.mode,
            'sigma': self.sigma,
            'kernel_size': self.kernel_size
        }

    def make_accumulate_fn(self):
        """Create the accumulation function based on mode."""
        if self.mode == 'analytic':
            return make_accumulate_square(self.height, self.width,
                                         self.x0, self.y0, self.dx, self.dy)
        else:
            return make_accumulate_square_splats(self.height, self.width,
                                                self.sigma, self.kernel_size,
                                                self.x0, self.y0, self.dx, self.dy)
