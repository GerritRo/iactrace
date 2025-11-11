"""Compiled simulation - pure JAX pytree for efficient rendering."""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
from ..core.geometry import intersect_plane, check_occlusions_batch
from ..core.reflection import reflect


class CompiledSimulation(NamedTuple):
    """
    Pure JAX pytree containing all telescope data for simulation.

    This is the output of Telescope.compile() - contains only arrays and
    a JIT-compiled render function. No Python overhead, fully composable.
    """
    # Mirror data
    mirror_points: jnp.ndarray       # (n_mirrors, n_samples, 3)
    mirror_normals: jnp.ndarray      # (n_mirrors, n_samples, 3)
    mirror_positions: jnp.ndarray    # (n_mirrors, 3)
    mirror_rotations: jnp.ndarray    # (n_mirrors, 3, 3)

    # Obstruction data
    cyl_p1: jnp.ndarray             # (n_cyl, 3)
    cyl_p2: jnp.ndarray             # (n_cyl, 3)
    cyl_radius: jnp.ndarray         # (n_cyl,)
    box_p1: jnp.ndarray             # (n_box, 3)
    box_p2: jnp.ndarray             # (n_box, 3)

    # Sensor configuration
    sensor_plane_pos: jnp.ndarray    # (3,)
    sensor_plane_normal: jnp.ndarray # (3,)
    sensor_config: dict              # Sensor parameters
    accumulate_fn: Callable          # Accumulation function

    # Source type
    source_type: str                 # 'point' or 'infinity'

    def __call__(self, source_input):
        """
        Render the telescope simulation.

        Args:
            source_input:
                - For 'point': single position array (3,) or batch (N, 3)
                - For 'infinity': single direction (3,) or batch (N, 3)

        Returns:
            Rendered image (sensor-dependent shape)
        """
        # Expand single source to batch for consistent processing
        if source_input.ndim == 1:
            source_input = source_input[None, :]  # (1, 3)

        return _render_kernel(self, source_input)


@jax.jit
def _render_kernel(sim: CompiledSimulation, sources):
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
        if sim.source_type == 'point':
            # dirs: (N_sources, n_samples, 3)
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        elif sim.source_type == 'infinity':
            # Broadcast directions to all samples
            dirs = jnp.broadcast_to(sources[:, None, :],
                                   (sources.shape[0], tp_single.shape[0], 3))
        else:
            raise ValueError(f"Unknown source type: {sim.source_type}")

        # Check occlusions for ALL sources
        has_cylinders = len(sim.cyl_p1) > 0
        has_boxes = len(sim.box_p1) > 0

        if has_cylinders or has_boxes:
            box_p1 = sim.box_p1 if has_boxes else None
            box_p2 = sim.box_p2 if has_boxes else None
            # tp_single is (n_samples, 3), broadcast to (N_sources, n_samples, 3)
            tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
            shadow_mask = check_occlusions_batch(tp_broadcast, -dirs,
                                                 sim.cyl_p1, sim.cyl_p2, sim.cyl_radius,
                                                 box_p1, box_p2)
        else:
            shadow_mask = jnp.ones((sources.shape[0], tp_single.shape[0]))

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
        acc0 = jnp.zeros(len(sim.sensor_config['hex_centers']))
    else:
        raise ValueError(f"Unknown sensor type: {sim.sensor_config['type']}")

    # Scan over all mirrors, accumulating contributions
    n_mirrors = sim.mirror_points.shape[0]
    final_img, _ = jax.lax.scan(render_single_mirror, acc0, jnp.arange(n_mirrors))

    return final_img


def _render_per_mirror(sim, source_input, mirror_idx):
    """
    Render contribution from a single mirror (useful for debugging/visualization).

    Args:
        sim: CompiledSimulation
        source_input: Source position or direction
        mirror_idx: Index of mirror to render

    Returns:
        Image from single mirror
    """
    # Transform points/normals for single mirror
    tp = jnp.einsum('ij,nj->ni', sim.mirror_rotations[mirror_idx],
                   sim.mirror_points[mirror_idx]) + sim.mirror_positions[mirror_idx]
    tn = jnp.einsum('ij,nj->ni', sim.mirror_rotations[mirror_idx],
                   sim.mirror_normals[mirror_idx])

    # Compute directions
    if sim.source_type == 'point':
        dirs = tp - source_input
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    else:
        dirs = jnp.broadcast_to(source_input, tp.shape)

    # Check occlusions
    has_cylinders = len(sim.cyl_p1) > 0
    has_boxes = len(sim.box_centers) > 0

    if has_cylinders or has_boxes:
        box_centers = sim.box_centers if has_boxes else None
        box_sizes = sim.box_sizes if has_boxes else None
        shadow_mask = check_occlusions_batch(tp[None, :, :], -dirs[None, :, :],
                                             sim.cyl_p1, sim.cyl_p2, sim.cyl_radius,
                                             box_centers, box_sizes)[0]
    else:
        shadow_mask = jnp.ones(tp.shape[0])

    # Reflect
    reflected, cos_angle = jax.vmap(reflect, in_axes=(0, 0))(dirs, tn)

    # Intersect
    pts = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
        tp, reflected, sim.sensor_plane_pos, sim.sensor_plane_normal
    )

    # Accumulate
    values = cos_angle[:, 0] * shadow_mask
    return sim.accumulate_fn(pts[:, 0], pts[:, 1], values)
