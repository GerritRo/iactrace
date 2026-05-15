from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..core.ray_bundle import RayBundle


class Concentrator(eqx.Module):
    """Abstract base for light concentrators."""

    @abstractmethod
    def trace_rays_batch(
        self, x: Array, y: Array, dx: Array, dy: Array, dz: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        ...

    @abstractmethod
    def compute_throughput_batch(
        self, x: Array, y: Array, dx: Array, dy: Array, dz: Array
    ) -> Array:
        ...

    def apply(self, ray_bundle: RayBundle) -> RayBundle:
        """Trace rays through concentrator, returning a modified RayBundle.

        Uses :meth:`trace_rays_batch` for full exit-ray state: updated
        origins, directions, and throughput-scaled values.
        """
        x, y = ray_bundle.origins[:, 0], ray_bundle.origins[:, 1]
        dx, dy, dz = (ray_bundle.directions[:, i] for i in range(3))
        tp, ex, ey, edx, edy, edz = self.trace_rays_batch(x, y, dx, dy, dz)
        return RayBundle(
            origins=jnp.stack([ex, ey, jnp.zeros_like(ex)], axis=-1),
            directions=jnp.stack([edx, edy, edz], axis=-1),
            values=ray_bundle.values * tp,
            path_length=ray_bundle.path_length,
            n=ray_bundle.n,
        )
