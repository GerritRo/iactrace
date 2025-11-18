import jax
import jax.numpy as jnp

from functools import partial
from .geometry import intersect_plane, check_occlusions
from .reflection import reflect
from .transforms import euler_to_matrix
    
@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render(tel, sources, source_type, sensor_idx=0):
    """
    Core rendering function - processes ALL sources at once, loops over mirrors.
    
    Args:
        tel: Telescope object
        sources: Batch of source positions/directions (N_sources, 3)
        source_type: 'point' or 'infinity'
        sensor_idx: Id of sensor in telescope object

    Returns:
        Rendered image with all sources accumulated
    """
    # Get sensor for this index
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)
    
    # Get mirror rotation matrices:
    mirror_pos = tel.mirror_positions
    mirror_rot = jax.vmap(euler_to_matrix)(tel.mirror_rotations)
    
    # Pre-transform all mirror points and normals to world space
    tp_all = jnp.einsum('mij,mnj->mni', mirror_rot, tel.mirror_points) + mirror_pos[:, None, :]
    tn_all = jnp.einsum('mij,mnj->mni', mirror_rot, tel.mirror_normals)

    # Function to render contribution from one mirror for ALL sources
    def render_single_mirror(acc, mirror_idx):
        tp_single = tp_all[mirror_idx]  # (n_samples, 3)
        tn_single = tn_all[mirror_idx]  # (n_samples, 3)
        tv_single = tel.mirror_weights[mirror_idx]  # (n_samples, 1)

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
            raise ValueError(f"Unknown source type: {source_type}")

        # tp_single is (n_samples, 3), broadcast to (N_sources, n_samples, 3)
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow_mask = check_occlusions(tp_broadcast, -dirs,
                                       tel.cyl_p1, tel.cyl_p2, tel.cyl_r,
                                       tel.box_p1, tel.box_p2)

        # Reflect rays off mirror (vmap over sources, then over samples)
        tn_broadcast = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        reflected, cos_angle = jax.vmap(jax.vmap(reflect, in_axes=(0, 0)),
                                       in_axes=(0, 0))(dirs, tn_broadcast)

        # Intersect with sensor plane
        tp_broadcast = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        pts = jax.vmap(jax.vmap(intersect_plane, in_axes=(0, 0, None, None)),
                      in_axes=(0, 0, None, None))(
            tp_broadcast, reflected, sensor_pos, sensor_rot
        )

        # Flatten all sources and samples for this mirror
        pts_flat = pts.reshape(-1, 2)
        values_flat = (cos_angle[..., 0] / tv_single[..., 0] * shadow_mask).flatten()

        # Accumulate contribution from this mirror
        img = sensor.accumulate(pts_flat[:, 0], pts_flat[:, 1], values_flat)

        return acc + img, None

    # Get initial accumulator based on sensor type
    acc0 = jnp.zeros(sensor.get_accumulator_shape())

    # Scan over all mirrors, accumulating contributions
    n_mirrors = tel.mirror_points.shape[0]
    final_img, _ = jax.lax.scan(render_single_mirror, acc0, jnp.arange(n_mirrors))

    return final_img

@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_debug(tel, sources, source_type, sensor_idx=0):
    """
    Renders without accumulating on sensor, instead returns raw hits.

    Returns:
        pts_all : (N_total, 2)
        vals_all: (N_total,)
    """
    sensor = tel.sensors[sensor_idx]
    sensor_pos = sensor.position
    sensor_rot = euler_to_matrix(sensor.rotation)

    mirror_pos = tel.mirror_positions
    mirror_rot = jax.vmap(euler_to_matrix)(tel.mirror_rotations)

    tp_all = jnp.einsum('mij,mnj->mni', mirror_rot, tel.mirror_points) + mirror_pos[:, None, :]
    tn_all = jnp.einsum('mij,mnj->mni', mirror_rot, tel.mirror_normals)

    def render_single_mirror(carry, mirror_idx):
        tp_single = tp_all[mirror_idx]
        tn_single = tn_all[mirror_idx]
        tv_single = tel.mirror_weights[mirror_idx]  # (n_samples, 1)

        # directions
        if source_type == 'point':
            dirs = tp_single[None, :, :] - sources[:, None, :]
            dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        else:
            dirs = jnp.broadcast_to(
                sources[:, None, :],
                (sources.shape[0], tp_single.shape[0], 3)
            )

        # occlusion
        tp_b = jnp.broadcast_to(tp_single[None, :, :], dirs.shape)
        shadow_mask = check_occlusions(
            tp_b, -dirs,
            tel.cyl_p1, tel.cyl_p2, tel.cyl_r,
            tel.box_p1, tel.box_p2
        )

        # reflect
        tn_b = jnp.broadcast_to(tn_single[None, :, :], dirs.shape)
        refl, cosang = jax.vmap(jax.vmap(reflect))(dirs, tn_b)

        # intersect
        pts = jax.vmap(jax.vmap(intersect_plane, in_axes=(0,0,None,None)),
                       in_axes=(0,0,None,None))(tp_b, refl, sensor_pos, sensor_rot)

        pts_flat = pts.reshape(-1, 2)
        vals_flat = (cosang[..., 0] / tv_single[..., 0] * shadow_mask).flatten()

        # carry stays the same (None), output is the data
        return carry, (pts_flat, vals_flat)

    # run scan
    _, per_mirror = jax.lax.scan(
        render_single_mirror,
        None,
        jnp.arange(tel.mirror_points.shape[0])
    )

    # per_mirror = (pts_list, vals_list)
    pts_all = jnp.concatenate(per_mirror[0], axis=0)
    vals_all = jnp.concatenate(per_mirror[1], axis=0)

    return pts_all, vals_all