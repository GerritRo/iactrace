import math

import jax
import jax.numpy as jnp
from functools import partial

from .intersections import intersect_plane
from .reflection import reflect
from .transforms import euler_to_matrix


def _get_stages(mirror_groups):
    """Group mirror_groups by optical_stage, return sorted dict."""
    from collections import defaultdict
    by_stage = defaultdict(list)
    for g in mirror_groups:
        by_stage[g.optical_stage].append(g)
    return dict(sorted(by_stage.items()))


def _check_occlusions(ray_origins, ray_directions, obstruction_groups):
    """
    Check ray occlusions against obstruction groups.

    Args:
        ray_origins: Ray origins (n_sources, n_samples, 3)
        ray_directions: Ray directions (n_sources, n_samples, 3)
        obstruction_groups: List of ObstructionGroup objects (can be empty or None)

    Returns:
        Shadow mask (n_sources, n_samples) - 1.0 if not occluded, 0.0 if occluded
    """
    # Handle empty or None obstruction_groups
    if not obstruction_groups:
        return jnp.ones(ray_origins.shape[:-1])
    
    shadow_mask = jnp.ones(ray_origins.shape[:-1])
    for group in obstruction_groups:
        t = jax.vmap(jax.vmap(group.intersect))(ray_origins, ray_directions)
        shadow_mask = shadow_mask * jnp.where(t < 1e10, 0.0, 1.0)
    return shadow_mask


def _reflect_at_stage(origins, directions, values, stage_groups, obstruction_groups):
    """
    Reflect 2D batch of rays off stage mirrors.

    Args:
        origins: (n_sources, n_samples, 3)
        directions: (n_sources, n_samples, 3)
        values: (n_sources, n_samples)

    Returns:
        new_origins, new_directions, new_values (same shapes)
    """
    n_sources, n_samples = origins.shape[:2]

    best_t = jnp.full((n_sources, n_samples), jnp.inf)
    best_points = jnp.zeros((n_sources, n_samples, 3))
    best_normals = jnp.zeros((n_sources, n_samples, 3))

    for group in stage_groups:
        # vmap _intersect_group over source dimension
        t, points, normals = jax.vmap(
            lambda o, d: _intersect_group(o, d, group)
        )(origins, directions)

        closer = t < best_t
        best_t = jnp.where(closer, t, best_t)
        best_points = jnp.where(closer[..., None], points, best_points)
        best_normals = jnp.where(closer[..., None], normals, best_normals)

    reflected, cos_angle = jax.vmap(jax.vmap(reflect))(directions, best_normals)

    hit_mask = best_t < 1e10
    shadow = _check_occlusions(origins, directions, obstruction_groups)
    new_values = values * hit_mask * shadow * jnp.abs(cos_angle[..., 0])

    return best_points, reflected, new_values


def _intersect_group(ray_origins, ray_directions, group):
    """Intersect flat rays (N, 3) with all mirrors in group, return closest hit."""
    surface = group.get_surface()
    n_rays = ray_origins.shape[0]

    def intersect_mirror(mirror_idx):
        pos = group.positions[mirror_idx]
        rot = euler_to_matrix(group.rotations[mirror_idx])
        rot_inv = rot.T
        offset = group.offsets[mirror_idx]

        o_local = jnp.einsum('ij,nj->ni', rot_inv, ray_origins - pos)
        d_local = jnp.einsum('ij,nj->ni', rot_inv, ray_directions)

        def intersect_one(o, d):
            return surface.intersect(o, d, offset)
        ts, pts_local, norms_local = jax.vmap(intersect_one)(o_local, d_local)

        in_aperture = group.check_aperture(pts_local[:, 0], pts_local[:, 1], mirror_idx)
        ts = jnp.where(in_aperture, ts, jnp.inf)

        pts_world = jnp.einsum('ij,nj->ni', rot, pts_local) + pos
        norms_world = jnp.einsum('ij,nj->ni', rot, norms_local)

        return ts, pts_world, norms_world

    all_ts, all_pts, all_norms = jax.vmap(intersect_mirror)(jnp.arange(len(group)))

    closest = jnp.argmin(all_ts, axis=0)
    best_t = jnp.min(all_ts, axis=0)
    best_pts = all_pts[closest, jnp.arange(n_rays)]
    best_norms = all_norms[closest, jnp.arange(n_rays)]

    return best_t, best_pts, best_norms


def _trace_single_mirror(mirror_idx, tp_all, tn_all, tw_all, sources, values,
                         source_type, n_sources, n_samples, stage_indices, stages,
                         obstruction_groups, sensor_pos, sensor_rot):
    """Trace rays through one primary mirror. Returns (pts, ray_vals).

    This is the common ray tracing logic extracted for reuse.
    """
    tp_single = tp_all[mirror_idx]
    tn_single = tn_all[mirror_idx]
    tw_single = tw_all[mirror_idx]

    if source_type == 'point':
        dirs = tp_single[None, :, :] - sources[:, None, :]
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    else:
        dirs = jnp.broadcast_to(sources[:, None, :], (n_sources, n_samples, 3))

    origins = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
    normals = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)

    shadow = _check_occlusions(origins, -dirs, obstruction_groups)

    reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, normals)
    ray_vals = values[:, None] * cos_angle[..., 0] / tw_single[None, :, 0] * shadow

    # Propagate through additional optical stages
    origins_cur, reflected_cur, ray_vals_cur = origins, reflected, ray_vals
    for stage_idx in stage_indices[1:]:
        origins_cur, reflected_cur, ray_vals_cur = _reflect_at_stage(
            origins_cur, reflected_cur, ray_vals_cur,
            stages[stage_idx], obstruction_groups
        )

    # Intersect with sensor plane
    pts = jax.vmap(
        jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
        in_axes=(0, 0, None, None)
    )(origins_cur, reflected_cur, sensor_pos, sensor_rot)

    return pts, ray_vals_cur  # (n_sources, n_samples, 2), (n_sources, n_samples)


def _accumulate_image(pts, ray_vals, sensor):
    """Accumulate all rays into single image (for render)."""
    pts_flat = pts.reshape(-1, 2)
    vals_flat = ray_vals.reshape(-1)
    return sensor.accumulate(pts_flat[:, 0], pts_flat[:, 1], vals_flat)


def _accumulate_per_source(pts, ray_vals, sensor):
    """Accumulate rays per-source into response matrix row (for render_response_matrix)."""
    return jax.vmap(
        lambda p, v: sensor.accumulate(p[:, 0], p[:, 1], v).reshape(-1)
    )(pts, ray_vals)


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render(tel, sources, values, source_type, sensor_idx=0):
    """Render sources through telescope onto sensor.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor to use

    Returns:
        Accumulated image with sensor shape.
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())

    n_sources = sources.shape[0]

    # Handle empty telescope
    if not stage_indices or 0 not in stages:
        return jnp.zeros(sensor.get_accumulator_shape())

    # Get primary mirror data
    group_data = [g.transform_to_world() for g in stages[0]]
    tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
    tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
    tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)

    n_mirrors = tp_all.shape[0]
    n_samples = tp_all.shape[1]

    def process_mirror(acc, mirror_idx):
        pts, ray_vals = _trace_single_mirror(
            mirror_idx, tp_all, tn_all, tw_all, sources, values,
            source_type, n_sources, n_samples, stage_indices, stages,
            tel.obstruction_groups, sensor_pos, sensor_rot
        )
        return acc + _accumulate_image(pts, ray_vals, sensor), None

    acc0 = jnp.zeros(sensor.get_accumulator_shape())
    result, _ = jax.lax.scan(process_mirror, acc0, jnp.arange(n_mirrors))
    return result


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_debug(tel, sources, values, source_type, sensor_idx=0):
    """Render without accumulation - returns raw hits.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor to use

    Returns:
        Tuple of (points, values) arrays with all ray intersections.
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())

    n_sources = sources.shape[0]

    # Handle empty telescope
    if not stage_indices or 0 not in stages:
        return jnp.zeros((0, 2)), jnp.zeros((0,))

    # Get primary mirror data
    group_data = [g.transform_to_world() for g in stages[0]]
    tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
    tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
    tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)

    n_mirrors = tp_all.shape[0]
    n_samples = tp_all.shape[1]

    def process_mirror(carry, mirror_idx):
        pts, ray_vals = _trace_single_mirror(
            mirror_idx, tp_all, tn_all, tw_all, sources, values,
            source_type, n_sources, n_samples, stage_indices, stages,
            tel.obstruction_groups, sensor_pos, sensor_rot
        )
        return carry, (pts.reshape(-1, 2), ray_vals.reshape(-1))

    _, per_mirror = jax.lax.scan(process_mirror, None, jnp.arange(n_mirrors))
    return per_mirror[0].reshape(-1, 2), per_mirror[1].reshape(-1)


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_response_matrix(tel, sources, values, source_type, sensor_idx=0):
    """Render multiple sources and return the source-to-pixel response matrix.

    This function traces N sources through the telescope and returns an N×M matrix
    where each row contains one source's contribution to all M pixels.
    Uses incremental accumulation for memory efficiency.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor to use

    Returns:
        Array of shape (n_sources, n_pixels) where n_pixels is the flattened sensor size.
        For square sensors: n_pixels = height * width
        For hexagonal sensors: n_pixels = n_hex_pixels
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())

    n_sources = sources.shape[0]
    n_pixels = math.prod(sensor.get_accumulator_shape())

    # Handle empty telescope
    if not stage_indices or 0 not in stages:
        return jnp.zeros((n_sources, n_pixels))

    # Get primary mirror data
    group_data = [g.transform_to_world() for g in stages[0]]
    tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
    tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
    tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)

    n_mirrors = tp_all.shape[0]
    n_samples = tp_all.shape[1]

    def process_mirror(acc, mirror_idx):
        pts, ray_vals = _trace_single_mirror(
            mirror_idx, tp_all, tn_all, tw_all, sources, values,
            source_type, n_sources, n_samples, stage_indices, stages,
            tel.obstruction_groups, sensor_pos, sensor_rot
        )
        return acc + _accumulate_per_source(pts, ray_vals, sensor), None

    acc0 = jnp.zeros((n_sources, n_pixels))
    result, _ = jax.lax.scan(process_mirror, acc0, jnp.arange(n_mirrors))
    return result