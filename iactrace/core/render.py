import jax
import jax.numpy as jnp
from functools import partial

from .geometry import intersect_plane
from .reflection import reflect
from .transforms import euler_to_matrix


def _get_stages(mirror_groups):
    """Group mirror_groups by optical_stage, return sorted dict."""
    from collections import defaultdict
    by_stage = defaultdict(list)
    for g in mirror_groups:
        by_stage[g.optical_stage].append(g)
    return dict(sorted(by_stage.items()))


def _check_occlusions(origins, directions, obstruction_groups):
    """Check occlusions for batch of rays (N, 3)."""
    shadow = jnp.ones(origins.shape[0])
    for group in obstruction_groups:
        t = jax.vmap(group.intersect)(origins, directions)
        shadow = shadow * jnp.where(t < 1e10, 0.0, 1.0)
    return shadow


def _reflect_at_stage(ray_origins, ray_directions, values, stage_groups, obstruction_groups):
    """Intersect rays with stage mirrors, reflect, apply occlusion mask."""
    n_rays = ray_origins.shape[0]
    
    best_t = jnp.full((n_rays,), jnp.inf)
    best_points = jnp.zeros((n_rays, 3))
    best_normals = jnp.zeros((n_rays, 3))
    
    for group in stage_groups:
        t, points, normals = _intersect_group(ray_origins, ray_directions, group)
        closer = t < best_t
        best_t = jnp.where(closer, t, best_t)
        best_points = jnp.where(closer[:, None], points, best_points)
        best_normals = jnp.where(closer[:, None], normals, best_normals)
    
    reflected, cos_angle = jax.vmap(reflect)(ray_directions, best_normals)
    
    hit_mask = best_t < 1e10
    shadow = _check_occlusions(ray_origins, ray_directions, obstruction_groups)
    new_values = values * hit_mask * shadow * jnp.abs(cos_angle[:, 0])
    
    return best_points, reflected, new_values


def _intersect_group(ray_origins, ray_directions, group):
    """Intersect rays with all mirrors in group, return closest hit."""
    surface = group.get_surface()
    n_rays = ray_origins.shape[0]
    
    def intersect_mirror(mirror_idx):
        pos = group.positions[mirror_idx]
        rot = euler_to_matrix(group.rotations[mirror_idx])
        rot_inv = rot.T
        offset = group.offsets[mirror_idx]
        
        # To local frame
        o_local = jnp.einsum('ij,nj->ni', rot_inv, ray_origins - pos)
        d_local = jnp.einsum('ij,nj->ni', rot_inv, ray_directions)
        
        # Intersect
        def intersect_one(o, d):
            return surface.intersect(o, d, offset)
        ts, pts_local, norms_local = jax.vmap(intersect_one)(o_local, d_local)
        
        # Aperture check
        in_aperture = group.check_aperture(pts_local[:, 0], pts_local[:, 1], mirror_idx)
        ts = jnp.where(in_aperture, ts, jnp.inf)
        
        # To world frame
        pts_world = jnp.einsum('ij,nj->ni', rot, pts_local) + pos
        norms_world = jnp.einsum('ij,nj->ni', rot, norms_local)
        
        return ts, pts_world, norms_world
    
    all_ts, all_pts, all_norms = jax.vmap(intersect_mirror)(jnp.arange(len(group)))
    
    closest = jnp.argmin(all_ts, axis=0)
    best_t = jnp.min(all_ts, axis=0)
    best_pts = all_pts[closest, jnp.arange(n_rays)]
    best_norms = all_norms[closest, jnp.arange(n_rays)]
    
    return best_t, best_pts, best_norms


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render(tel, sources, values, source_type, sensor_idx=0):
    """Render sources through telescope onto sensor."""
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)
    
    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())
    
    if not stage_indices or 0 not in stages:
        return jnp.zeros(sensor.get_accumulator_shape())
    
    # Get stage 0 data
    group_data = [g.transform_to_world() for g in stages[0]]
    tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
    tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
    tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)
    
    n_mirrors = tp_all.shape[0]
    n_samples = tp_all.shape[1]
    n_sources = sources.shape[0]
    
    def process_mirror(acc, mirror_idx):
        tp_single = tp_all[mirror_idx]
        tn_single = tn_all[mirror_idx]
        tw_single = tw_all[mirror_idx]
        
        # Compute directions to primary
        if source_type == 'point':
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        else:
            dirs = jnp.broadcast_to(sources[:, None, :], (n_sources, n_samples, 3))
        
        # Check occlusions before primary
        tp_bc = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow = _check_occlusions(
            tp_bc.reshape(-1, 3), 
            -dirs.reshape(-1, 3), 
            tel.obstruction_groups
        ).reshape(dirs.shape[:-1])
        
        # Reflect off primary
        tn_bc = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, tn_bc)
        ray_vals = values[:, None] * cos_angle[..., 0] / tw_single[None, :, 0] * shadow
        
        # Flatten for subsequent stages
        flat_origins = tp_bc.reshape(-1, 3)
        flat_dirs = reflected.reshape(-1, 3)
        flat_values = ray_vals.reshape(-1)
        
        # Process through stages 1+
        for stage_idx in stage_indices[1:]:
            flat_origins, flat_dirs, flat_values = _reflect_at_stage(
                flat_origins, flat_dirs, flat_values,
                stages[stage_idx], tel.obstruction_groups
            )
        
        # Intersect with sensor
        pts, t = jax.vmap(
            lambda o, d: intersect_plane(o, d, sensor_pos, sensor_rot)
        )(flat_origins, flat_dirs)
        
        forward_mask = jnp.where(t > 0, 1.0, 0.0)
        
        # Accumulate
        img = sensor.accumulate(pts[:, 0], pts[:, 1], flat_values*forward_mask)
        return acc + img, None
    
    acc0 = jnp.zeros(sensor.get_accumulator_shape())
    final_img, _ = jax.lax.scan(process_mirror, acc0, jnp.arange(n_mirrors))
    
    return final_img


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_debug(tel, sources, values, source_type, sensor_idx=0):
    """Render without accumulation - returns raw hits."""
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)
    
    stages = _get_stages(tel.mirror_groups)
    stage_indices = sorted(stages.keys())
    
    if not stage_indices or 0 not in stages:
        return jnp.zeros((0, 2)), jnp.zeros((0,))
    
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
        
        tp_bc = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow = _check_occlusions(
            tp_bc.reshape(-1, 3), 
            -dirs.reshape(-1, 3), 
            tel.obstruction_groups
        ).reshape(dirs.shape[:-1])
        
        tn_bc = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, tn_bc)
        ray_vals = values[:, None] * cos_angle[..., 0] / tw_single[None, :, 0] * shadow
        
        flat_origins = tp_bc.reshape(-1, 3)
        flat_dirs = reflected.reshape(-1, 3)
        flat_values = ray_vals.reshape(-1)
        
        for stage_idx in stage_indices[1:]:
            flat_origins, flat_dirs, flat_values = _reflect_at_stage(
                flat_origins, flat_dirs, flat_values,
                stages[stage_idx], tel.obstruction_groups
            )
        
        pts, t = jax.vmap(
            lambda o, d: intersect_plane(o, d, sensor_pos, sensor_rot)
        )(flat_origins, flat_dirs)
        forward_mask = jnp.where(t > 0, 1.0, 0.0)
        
        return carry, (pts, flat_values*forward_mask)
    
    _, per_mirror = jax.lax.scan(process_mirror, None, jnp.arange(n_mirrors))
    
    pts_all = per_mirror[0].reshape(-1, 2)
    vals_all = per_mirror[1].reshape(-1)
    
    return pts_all, vals_all