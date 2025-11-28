import jax
import jax.numpy as jnp
from functools import partial

from .geometry import intersect_plane
from .reflection import reflect
from .transforms import euler_to_matrix


def check_occlusions(ray_origins, ray_directions, obstruction_groups):
    """
    Check ray occlusions against obstruction groups.
    
    Args:
        ray_origins: Ray origins (..., 3)
        ray_directions: Ray directions (..., 3)
        obstruction_groups: List of ObstructionGroup objects
    
    Returns:
        Shadow mask (...) - 1.0 if not occluded, 0.0 if occluded
    """
    shadow_mask = jnp.ones(ray_origins.shape[:-1])
    
    # Loop over obstruction groups
    for group in obstruction_groups:
        t = jax.vmap(jax.vmap(group.intersect, in_axes=(0, 0)), in_axes=(0, 0))(
            ray_origins, ray_directions
        )
        shadow_mask = shadow_mask * jnp.where(t < 1e10, 0.0, 1.0)
    
    return shadow_mask


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render(tel, sources, values, source_type, sensor_idx=0):
    """
    Core rendering function.
    
    Args:
        tel: Telescope object
        sources: Source positions/directions (N_sources, 3)
        values: Flux values (N_sources,)
        source_type: 'point' or 'infinity'
        sensor_idx: Sensor index
    
    Returns:
        Rendered image
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    # Transform all mirror groups to world space at trace time
    group_data = [g.transform_to_world() for g in tel.mirror_groups]

    # Concatenate all groups to get (N_mirrors_total, M, 3) arrays
    if group_data:
        tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)  # (N_mirrors, M, 3)
        tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)  # (N_mirrors, M, 3)
        tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)  # (N_mirrors, M, 1)
    else:
        # Fallback to individual mirrors if no groups
        world_data = [m.transform_to_world() for m in tel.mirrors]
        tp_all = jnp.stack([d[0] for d in world_data])  # (N_mirrors, M, 3)
        tn_all = jnp.stack([d[1] for d in world_data])  # (N_mirrors, M, 3)
        tw_all = jnp.stack([d[2] for d in world_data])  # (N_mirrors, M, 1)
    
    def render_single_mirror(acc, mirror_idx):
        tp_single = tp_all[mirror_idx]
        tn_single = tn_all[mirror_idx]
        tw_single = tw_all[mirror_idx]
        
        # Compute ray directions
        if source_type == 'point':
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        elif source_type == 'infinity':
            dirs = jnp.broadcast_to(
                sources[:, None, :],
                (sources.shape[0], tp_single.shape[0], 3)
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Check occlusions
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow_mask = check_occlusions(tp_broadcast, -dirs, tel.obstruction_groups)
        
        # Reflect off mirror
        tn_broadcast = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, tn_broadcast)
        
        # Intersect with sensor plane
        pts = jax.vmap(
            jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None)
        )(tp_broadcast, reflected, sensor_pos, sensor_rot)
        
        # Accumulate
        pts_flat = pts.reshape(-1, 2)
        values_flat = (
            values[:, None] * cos_angle[..., 0] / tw_single[..., 0] * shadow_mask
        ).flatten()
        
        img = sensor.accumulate(pts_flat[:, 0], pts_flat[:, 1], values_flat)
        return acc + img, None
    
    acc0 = jnp.zeros(sensor.get_accumulator_shape())
    n_mirrors = tp_all.shape[0]  # Total number of mirrors across all groups
    final_img, _ = jax.lax.scan(render_single_mirror, acc0, jnp.arange(n_mirrors))
    
    return final_img


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_debug(tel, sources, values, source_type, sensor_idx=0):
    """
    Render without accumulating - returns raw hit points and values.
    
    Returns:
        pts_all: (N_total, 2)
        vals_all: (N_total,)
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    # Transform all mirror groups to world space at trace time
    group_data = [g.transform_to_world() for g in tel.mirror_groups]

    # Concatenate all groups to get (N_mirrors_total, M, 3) arrays
    if group_data:
        tp_all = jnp.concatenate([d[0] for d in group_data], axis=0)
        tn_all = jnp.concatenate([d[1] for d in group_data], axis=0)
        tw_all = jnp.concatenate([d[2] for d in group_data], axis=0)
    else:
        # Fallback to individual mirrors if no groups
        world_data = [m.transform_to_world() for m in tel.mirrors]
        tp_all = jnp.stack([d[0] for d in world_data])
        tn_all = jnp.stack([d[1] for d in world_data])
        tw_all = jnp.stack([d[2] for d in world_data])
    
    def render_single_mirror(carry, mirror_idx):
        tp_single = tp_all[mirror_idx]
        tn_single = tn_all[mirror_idx]
        tw_single = tw_all[mirror_idx]
        
        if source_type == 'point':
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        else:
            dirs = jnp.broadcast_to(
                sources[:, None, :],
                (sources.shape[0], tp_single.shape[0], 3)
            )
        
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow_mask = check_occlusions(tp_broadcast, -dirs, tel.obstruction_groups)
        
        tn_broadcast = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect))(dirs, tn_broadcast)
        
        pts = jax.vmap(
            jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, None)
        )(tp_broadcast, reflected, sensor_pos, sensor_rot)
        
        pts_flat = pts.reshape(-1, 2)
        vals_flat = (
            values[:, None] * cos_angle[..., 0] / tw_single[..., 0] * shadow_mask
        ).flatten()
        
        return carry, (pts_flat, vals_flat)
    
    _, per_mirror = jax.lax.scan(
        render_single_mirror,
        None,
        jnp.arange(tp_all.shape[0])
    )
    
    pts_all = jnp.concatenate(per_mirror[0], axis=0)
    vals_all = jnp.concatenate(per_mirror[1], axis=0)
    
    return pts_all, vals_all