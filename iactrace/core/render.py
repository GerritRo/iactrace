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
        obstruction_groups: List of ObstructionGroup objects
    
    Returns:
        Shadow mask (n_sources, n_samples) - 1.0 if not occluded, 0.0 if occluded
    """
    shadow_mask = jnp.ones(ray_origins.shape[:-1])
    for group in obstruction_groups:
        t = jax.vmap(jax.vmap(group.intersect))(ray_origins, ray_directions)
        shadow_mask = shadow_mask * jnp.where(t < 1e10, 0.0, 1.0)
    return shadow_mask


def _reflect_at_stage(origins, directions, values, stage_groups, obstruction_groups, include_shadowing):
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
    shadow = _check_occlusions(origins, directions, obstruction_groups) if include_shadowing else 1.0
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


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx', 'return_debug', 'include_shadowing'])
def _render_core(tel, sources, values, source_type, sensor_idx=0, return_debug=False, include_shadowing=True):
    """Core rendering logic shared by render() and render_debug().

    Args:
        tel: Telescope object
        sources: Source positions or directions
        values: Source intensities
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor to use
        return_debug: If True, return raw hits; if False, return accumulated image

    Returns:
        If return_debug=False: accumulated image array
        If return_debug=True: tuple of (points, values) arrays
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())

    if not stage_indices or 0 not in stages:
        if return_debug:
            return jnp.zeros((0, 2)), jnp.zeros((0,))
        else:
            return jnp.zeros(sensor.get_accumulator_shape())

    group_data = [g.transform_to_world() for g in stages[0]]
    tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
    tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
    tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)

    n_mirrors = tp_all.shape[0]
    n_samples = tp_all.shape[1]
    n_sources = sources.shape[0]

    def process_mirror(carry, mirror_idx):
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

        shadow = _check_occlusions(origins, -dirs, tel.obstruction_groups) if include_shadowing else 1.0

        reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, normals)
        ray_vals = values[:, None] * cos_angle[..., 0] / tw_single[None, :, 0] * shadow

        for stage_idx in stage_indices[1:]:
            origins, reflected, ray_vals = _reflect_at_stage(
                origins, reflected, ray_vals,
                stages[stage_idx], tel.obstruction_groups, include_shadowing
            )

        pts = jax.vmap(
            jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None)
        )(origins, reflected, sensor_pos, sensor_rot)

        pts_flat = pts.reshape(-1, 2)
        vals_flat = ray_vals.reshape(-1)

        if return_debug:
            return carry, (pts_flat, vals_flat)
        else:
            img = sensor.accumulate(pts_flat[:, 0], pts_flat[:, 1], vals_flat)
            return carry + img, None

    if return_debug:
        _, per_mirror = jax.lax.scan(process_mirror, None, jnp.arange(n_mirrors))
        pts_all = per_mirror[0].reshape(-1, 2)
        vals_all = per_mirror[1].reshape(-1)
        return pts_all, vals_all
    else:
        acc0 = jnp.zeros(sensor.get_accumulator_shape())
        final_img, _ = jax.lax.scan(process_mirror, acc0, jnp.arange(n_mirrors))
        return final_img


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx', 'include_shadowing'])
def render(tel, sources, values, source_type, sensor_idx=0, include_shadowing=True):
    """Render sources through telescope onto sensor."""
    return _render_core(tel, sources, values, source_type, sensor_idx, False, include_shadowing)


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx', 'include_shadowing'])
def render_debug(tel, sources, values, source_type, sensor_idx=0, include_shadowing=True):
    """Render without accumulation - returns raw hits."""
    return _render_core(tel, sources, values, source_type, sensor_idx, True, include_shadowing)