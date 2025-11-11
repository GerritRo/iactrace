"""Compiled simulation - pure JAX pytree for efficient rendering."""

import jax
import equinox as eqx
import jax.numpy as jnp
from jax import Array

from typing import Callable
from functools import partial
from ..core.geometry import intersect_plane, check_occlusions
from ..core.reflection import reflect

class CompiledSimulation(eqx.Module):
    """
    Pure JAX pytree containing all telescope data for simulation.
    This is the output of Telescope.compile() - contains only arrays and
    a JIT-compiled render function. No Python overhead, fully composable.
    """
    # Mirror data
    mirror_points: Array       # (n_mirrors, n_samples, 3)
    mirror_normals: Array      # (n_mirrors, n_samples, 3)
    mirror_positions: Array    # (n_mirrors, 3)
    mirror_rotations: Array    # (n_mirrors, 3, 3)
    
    # Obstruction data
    cyl_p1: Array             # (n_cyl, 3)
    cyl_p2: Array             # (n_cyl, 3)
    cyl_radius: Array         # (n_cyl,)
    box_p1: Array             # (n_box, 3)
    box_p2: Array             # (n_box, 3)
    
    # Sensor configuration
    sensor_plane_pos: Array    # (3,)
    sensor_plane_normal: Array # (3,)
    sensor_config: dict = eqx.field(static=True)
    accumulate_fn: Callable = eqx.field(static=True)
    
    def __call__(self, source_input, source_type, **overrides):
        """
        Args:
            source_input: source positions/directions
            **overrides: mirror_normals=new_normals, etc.
        """
        if source_input.ndim == 1:
            source_input = source_input[None, :]
        
        sim = replace(self, **overrides) if overrides else self
        return _render_kernel(sim, source_input, source_type)

    
@partial(jax.jit, static_argnames=['source_type'])
def _render_kernel(sim: CompiledSimulation , sources, source_type):
    """
    Core rendering kernel - processes ALL sources at once, loops over mirrors.

    This matches the notebook pattern: vmap over sources, scan over mirrors.
    Much faster than vmapping over sources separately.

    Args:
        sim: CompiledSimulation pytree
        sources: Batch of source positions/directions (N_sources, 3)

    Returns:
        Rendered image with all sources accumulated
    """
    # Pre-transform all mirror points and normals to world space
    tp_all = jnp.einsum('mij,mnj->mni', sim.mirror_rotations, sim.mirror_points) + \
             sim.mirror_positions[:, None, :]
    tn_all = jnp.einsum('mij,mnj->mni', sim.mirror_rotations, sim.mirror_normals)

    # Function to render contribution from one mirror for ALL sources
    def render_single_mirror(acc, mirror_idx):
        tp_single = tp_all[mirror_idx]  # (n_samples, 3)
        tn_single = tn_all[mirror_idx]  # (n_samples, 3)

        # Compute directions for ALL sources at once
        if source_type == 'point':
            # dirs: (N_sources, n_samples, 3)
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        elif source_type == 'infinity':
            # Broadcast directions to all samples
            dirs = jnp.broadcast_to(sources[:, None, :],
                                   (sources.shape[0], tp_single.shape[0], 3))
        else:
            raise ValueError(f"Unknown source type: {sim.source_type}")

        # tp_single is (n_samples, 3), broadcast to (N_sources, n_samples, 3)
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow_mask = check_occlusions(tp_broadcast, -dirs,
                                       sim.cyl_p1, sim.cyl_p2, sim.cyl_radius,
                                       sim.box_p1, sim.box_p2)

        # Reflect rays off mirror (vmap over sources, then over samples)
        tn_broadcast = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect, in_axes=(0, 0)),
                                       in_axes=(0, 0))(dirs, tn_broadcast)

        # Intersect with sensor plane
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        pts = jax.vmap(jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
                      in_axes=(0, 0, None, None))(
            tp_broadcast, reflected, sim.sensor_plane_pos, sim.sensor_plane_normal
        )

        # Flatten all sources and samples for this mirror
        pts_flat = pts.reshape(-1, 2)
        values_flat = (cos_angle[..., 0] * shadow_mask).flatten()

        # Accumulate contribution from this mirror
        img = sim.accumulate_fn(pts_flat[:, 0], pts_flat[:, 1], values_flat)

        return acc + img, None

    # Get initial accumulator based on sensor type
    if sim.sensor_config['type'] == 'square':
        acc0 = jnp.zeros((sim.sensor_config['height'], sim.sensor_config['width']))
    elif sim.sensor_config['type'] == 'hexagonal':
        acc0 = jnp.zeros(sim.sensor_config['hex_amount'])
    else:
        raise ValueError(f"Unknown sensor type: {sim.sensor_config['type']}")

    # Scan over all mirrors, accumulating contributions
    n_mirrors = sim.mirror_points.shape[0]
    final_img, _ = jax.lax.scan(render_single_mirror, acc0, jnp.arange(n_mirrors))

    return final_img