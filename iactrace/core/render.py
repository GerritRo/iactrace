import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from .intersections import intersect_plane
from .optics import (
    InteractionType,
    apply_perturbation,
    fresnel_unpolarized,
    reflect,
    refract,
    refract_slab,
)
from .transforms import euler_to_matrix


def _get_stages(optical_groups):
    """Group optical_groups by optical_stage, return sorted dict.

    Works with both mirror_groups and lens_groups (any OpticalGroupBase).
    """
    from collections import defaultdict
    by_stage = defaultdict(list)
    for g in optical_groups:
        by_stage[g.optical_stage].append(g)
    return dict(sorted(by_stage.items()))


def _check_occlusions(ray_origins, ray_directions, obstruction_groups, max_t=None):
    """
    Check ray occlusions against obstruction groups.

    Args:
        ray_origins: Ray origins (n_sources, n_samples, 3)
        ray_directions: Ray directions (n_sources, n_samples, 3)
        obstruction_groups: List of ObstructionGroup objects (can be empty or None)
        max_t: Optional maximum t value - obstructions at t >= max_t are ignored.
               Shape must broadcast with (n_sources, n_samples). If None, uses 1e10.

    Returns:
        Shadow mask (n_sources, n_samples) - 1.0 if not occluded, 0.0 if occluded
    """
    if not obstruction_groups:
        return jnp.ones(ray_origins.shape[:-1])

    if max_t is None:
        max_t = 1e10

    shadow_mask = jnp.ones(ray_origins.shape[:-1])
    for group in obstruction_groups:
        t = jax.vmap(jax.vmap(group.intersect))(ray_origins, ray_directions)
        shadow_mask = shadow_mask * jnp.where(t < max_t, 0.0, 1.0)
    return shadow_mask


def _intersect_optical_group(ray_origins, ray_directions, group):
    """Intersect flat rays (N, 3) with all elements in an optical group.

    Unified intersection function that works for mirrors, lenses, and slabs.
    Returns closest hit with all relevant data.

    Returns:
        t: (N,) ray parameters at intersection
        pts: (N, 3) world-space intersection points
        norms: (N, 3) world-space surface normals (with roughness applied)
        element_idx: (N,) index of closest element hit by each ray
    """
    n_rays = ray_origins.shape[0]

    def intersect_element(element_idx):
        surface = group.get_surface(element_idx)
        pos = group.positions[element_idx]
        rot = euler_to_matrix(group.rotations[element_idx])
        rot_inv = rot.T
        offset = group.offsets[element_idx]

        o_local = jnp.einsum('ij,nj->ni', rot_inv, ray_origins - pos)
        d_local = jnp.einsum('ij,nj->ni', rot_inv, ray_directions)

        def intersect_one(o, d):
            return surface.intersect(o, d, offset)
        ts, pts_local, norms_local = jax.vmap(intersect_one)(o_local, d_local)

        in_aperture = group.check_aperture(pts_local[:, 0], pts_local[:, 1], element_idx)
        ts = jnp.where(in_aperture, ts, jnp.inf)

        pts_world = jnp.einsum('ij,nj->ni', rot, pts_local) + pos
        norms_world = jnp.einsum('ij,nj->ni', rot, norms_local)

        # Apply surface roughness perturbation
        scale = group.perturbation_scale[element_idx]

        def get_angles(ray_idx):
            key = jr.fold_in(group.perturbation_key, element_idx)
            key = jr.fold_in(key, ray_idx)
            return jr.normal(key, (2,))

        angles = jax.vmap(get_angles)(jnp.arange(n_rays))
        norms_world = apply_perturbation(norms_world, angles, scale)

        return ts, pts_world, norms_world

    all_ts, all_pts, all_norms = jax.vmap(intersect_element)(jnp.arange(len(group)))

    closest = jnp.argmin(all_ts, axis=0)
    best_t = jnp.min(all_ts, axis=0)
    best_pts = all_pts[closest, jnp.arange(n_rays)]
    best_norms = all_norms[closest, jnp.arange(n_rays)]

    return best_t, best_pts, best_norms, closest


def _intersect_sensor_group(ray_origins, ray_directions, sensor_group):
    """Intersect rays with all sensors in a sensor group, find closest hit.

    Args:
        ray_origins: (n_rays, 3) ray origins
        ray_directions: (n_rays, 3) ray directions
        sensor_group: SensorGroup with positions (N, 3) and rotations (N, 3)

    Returns:
        pts: (n_rays, 2) local 2D coordinates on the hit sensor plane
        sensor_idx: (n_rays,) index of closest sensor hit
        best_t: (n_rays,) t parameter of closest hit
    """
    n_rays = ray_origins.shape[0]
    n_sensors = sensor_group.n_sensors

    def intersect_sensor(s_idx):
        pos = sensor_group.positions[s_idx]
        rot = euler_to_matrix(sensor_group.rotations[s_idx])

        # Intersect all rays with this sensor plane
        pts_t = jax.vmap(
            intersect_plane, in_axes=(0, 0, None, None)
        )(ray_origins, ray_directions, pos, rot)

        # intersect_plane returns (pts, t)
        pts = pts_t[0]  # (n_rays, 2)
        ts = pts_t[1]   # (n_rays,)

        return pts, ts

    # Intersect all sensors: all_pts (n_sensors, n_rays, 2), all_ts (n_sensors, n_rays)
    all_pts, all_ts = jax.vmap(intersect_sensor)(jnp.arange(n_sensors))

    # Find closest sensor for each ray
    closest_sensor = jnp.argmin(all_ts, axis=0)  # (n_rays,)
    best_t = jnp.min(all_ts, axis=0)  # (n_rays,)
    best_pts = all_pts[closest_sensor, jnp.arange(n_rays)]  # (n_rays, 2)

    return best_pts, closest_sensor.astype(jnp.int32), best_t


def _apply_interaction(directions, normals, points, group, element_idx, interaction):
    """Apply optical interaction physics based on interaction type.

    Args:
        directions: (N, 3) incident ray directions
        normals: (N, 3) surface normals at hit points
        points: (N, 3) hit positions (used for slab exit position)
        group: optical group (for material properties)
        element_idx: (N,) indices of hit elements
        interaction: InteractionType

    Returns:
        new_directions: (N, 3) outgoing ray directions
        new_positions: (N, 3) outgoing ray positions (different from points for slabs)
        coefficients: (N,) interaction coefficients (reflectivity or transmission)
    """
    if interaction == InteractionType.REFLECT:
        reflected, _ = jax.vmap(reflect)(directions, normals)
        coeffs = group.reflectivity[element_idx]
        return reflected, points, coeffs

    elif interaction == InteractionType.REFRACT:
        n_inside = group.n_inside[element_idx]
        transmittance = group.transmittance[element_idx]
        n_outside = group.n_outside

        def refract_single(d, n, n_in):
            refracted, cos_t, tir = refract(d, n, n_outside, n_in)
            cos_i = jnp.abs(jnp.dot(d, n))
            _, T = fresnel_unpolarized(cos_i, cos_t, n_outside, n_in)
            T = jnp.where(tir, 0.0, T)
            return refracted, T

        refracted, fresnel_T = jax.vmap(refract_single)(directions, normals, n_inside)
        coeffs = transmittance * fresnel_T
        return refracted, points, coeffs

    else:  # InteractionType.SLAB
        n_inside = group.n_inside[element_idx]
        transmittance = group.transmittance[element_idx]
        thickness = group.thickness[element_idx]
        n_outside = group.n_outside

        def refract_slab_single(d, n, pos, n_in, thick):
            exit_dir, exit_pos, trans_coeff, _ = refract_slab(d, n, pos, n_outside, n_in, thick)
            return exit_dir, exit_pos, trans_coeff

        exit_dirs, exit_positions, trans_coeffs = jax.vmap(refract_slab_single)(
            directions, normals, points, n_inside, thickness
        )
        coeffs = transmittance * trans_coeffs
        return exit_dirs, exit_positions, coeffs


def _interact_at_stage_rays(origins, directions, values, stage_groups, obstruction_groups):
    """
    Process 1D batch of rays through optical groups at a stage.

    Unified function that handles reflection, refraction, and slab interactions.

    Args:
        origins: (n_rays, 3) ray origins
        directions: (n_rays, 3) ray directions
        values: (n_rays,) ray intensities
        stage_groups: list of optical groups at this stage
        obstruction_groups: list of ObstructionGroup for shadow checks

    Returns:
        new_origins, new_directions, new_values (same shapes)
    """
    if not stage_groups:
        return origins, directions, values

    n_rays = origins.shape[0]
    interaction = stage_groups[0].interaction

    # Find closest intersection across all groups
    best_t = jnp.full((n_rays,), jnp.inf)
    best_points = jnp.zeros((n_rays, 3))
    best_normals = jnp.zeros((n_rays, 3))
    best_element_idx = jnp.zeros((n_rays,), dtype=jnp.int32)

    for group in stage_groups:
        t, points, normals, element_idx = _intersect_optical_group(origins, directions, group)

        closer = t < best_t
        best_t = jnp.where(closer, t, best_t)
        best_points = jnp.where(closer[:, None], points, best_points)
        best_normals = jnp.where(closer[:, None], normals, best_normals)
        best_element_idx = jnp.where(closer, element_idx, best_element_idx)

    # Apply interaction using first group's properties.
    # Note: All groups at a stage should have compatible material properties.
    group = stage_groups[0]
    new_dirs, new_origins, coeffs = _apply_interaction(
        directions, best_normals, best_points, group, best_element_idx, interaction
    )

    hit_mask = best_t < 1e10
    shadow = _check_occlusions(
        origins[None, :, :], directions[None, :, :], obstruction_groups, best_t[None, :]
    )[0]
    new_values = values * hit_mask * shadow * coeffs

    return new_origins, new_dirs, new_values




def _trace_rays(ray_origins, ray_directions, values, stages, stage_indices,
                obstruction_groups, sensor_group):
    """
    Trace classical rays through full optical system including stage 0.

    Args:
        ray_origins: (n_rays, 3) ray starting positions
        ray_directions: (n_rays, 3) ray directions (should be normalized)
        values: (n_rays,) ray intensities
        stages: dict mapping stage index to list of optical groups
        stage_indices: sorted list of stage indices
        obstruction_groups: list of ObstructionGroup
        sensor_group: SensorGroup for final accumulation

    Returns:
        pts: (n_rays, 2) sensor coordinates (in local sensor frame)
        sensor_idx: (n_rays,) index of sensor each ray hit
        ray_vals: (n_rays,) final ray values
    """
    origins_cur = ray_origins
    directions_cur = ray_directions
    values_cur = values

    for stage_idx in stage_indices:
        origins_cur, directions_cur, values_cur = _interact_at_stage_rays(
            origins_cur, directions_cur, values_cur,
            stages[stage_idx], obstruction_groups
        )

    pts, sensor_idx, _ = _intersect_sensor_group(
        origins_cur, directions_cur, sensor_group
    )

    return pts, sensor_idx, values_cur


def _get_stage0_data(stages):
    """Extract stage 0 sample points, normals, and weights."""
    group_data = [g.transform_to_world() for g in stages[0]]
    return {
        'points': jnp.concatenate([d[0] for d in group_data], axis=0),
        'normals': jnp.concatenate([d[1] for d in group_data], axis=0),
        'weights': jnp.concatenate([d[2] for d in group_data], axis=0),
    }


def _trace_single_element(element_idx, data, sources, values, source_type,
                          n_sources, n_samples, stage_indices, stages,
                          obstruction_groups, sensor_group):
    """Trace rays from one primary element. Returns flat (pts, sensor_idx, ray_vals).

    Stage 0 uses pre-sampled points (skips intersection). Stages 1+ use
    uniform ray tracing via _interact_at_stage_rays.
    """
    points = data['points'][element_idx]   # (n_samples, 3)
    normals = data['normals'][element_idx]  # (n_samples, 3)
    weights = data['weights'][element_idx]  # (n_samples, 1)

    # Compute ray directions based on source type
    if source_type == 'point':
        # (n_sources, n_samples, 3)
        dirs = points[None, :, :] - sources[:, None, :]
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    else:
        dirs = jnp.broadcast_to(sources[:, None, :], (n_sources, n_samples, 3))

    # Flatten immediately to 1D rays
    n_rays = n_sources * n_samples
    dirs_flat = dirs.reshape(n_rays, 3)
    origins_flat = jnp.tile(points, (n_sources, 1))
    normals_flat = jnp.tile(normals, (n_sources, 1))

    # Shadow check on incoming rays
    shadow = _check_occlusions(
        origins_flat[None], -dirs_flat[None], obstruction_groups
    )[0]

    # Initial values: source intensity / integration weight
    weights_flat = jnp.tile(weights[:, 0], n_sources)
    vals_flat = jnp.repeat(values, n_samples) / weights_flat * shadow

    # Stage 0 physics: use pre-sampled points (skip intersection)
    group = stages[0][0]
    interaction = group.interaction
    element_indices = jnp.full((n_rays,), element_idx, dtype=jnp.int32)

    new_dirs, new_origins, coeffs = _apply_interaction(
        dirs_flat, normals_flat, origins_flat, group, element_indices, interaction
    )

    # For REFLECT: use cos_angle (rendering equation) instead of reflectivity
    # Note: reflectivity is already incorporated in the integration weights
    if interaction == InteractionType.REFLECT:
        cos_angle = jnp.abs(jnp.sum(dirs_flat * normals_flat, axis=-1))
        coeffs = cos_angle

    vals_flat = vals_flat * coeffs

    # Stages 1+: uniform ray tracing
    for stage_idx in stage_indices[1:]:
        new_origins, new_dirs, vals_flat = _interact_at_stage_rays(
            new_origins, new_dirs, vals_flat,
            stages[stage_idx], obstruction_groups
        )

    # Intersect with sensor group (all sensors)
    pts, sensor_idx, _ = _intersect_sensor_group(new_origins, new_dirs, sensor_group)

    return pts, sensor_idx, vals_flat


def _accumulate_image(pts, sensor_idx, ray_vals, sensor, n_sources, n_samples):
    """Accumulate all rays into sensor group (for render)."""
    return sensor.accumulate(sensor_idx, pts[:, 0], pts[:, 1], ray_vals)


def _accumulate_per_source(pts, sensor_idx, ray_vals, sensor, n_sources, n_samples):
    """Accumulate rays per-source into response matrix row.

    Returns:
        (n_sources, n_sensors * n_pixels) response matrix rows
    """
    # Reshape flat arrays to (n_sources, n_samples) for per-source accumulation
    pts_2d = pts.reshape(n_sources, n_samples, 2)
    sensor_idx_2d = sensor_idx.reshape(n_sources, n_samples)
    vals_2d = ray_vals.reshape(n_sources, n_samples)

    def accumulate_one_source(p, s_idx, v):
        # Accumulate for one source, returns (n_sensors, *per_sensor_shape)
        result = sensor.accumulate(s_idx, p[:, 0], p[:, 1], v)
        return result.reshape(-1)  # Flatten to (n_sensors * n_pixels,)

    return jax.vmap(accumulate_one_source)(pts_2d, sensor_idx_2d, vals_2d)


def _render_generic(tel, sources, values, source_type, sensor_idx, accumulate_fn):
    """Generic render function that works for all interaction types.

    Args:
        tel: Telescope object
        sources: Source positions or directions
        values: Source intensities
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor group to use
        accumulate_fn: Function (pts, sensor_idx, ray_vals, sensor, n_sources, n_samples) -> result

    Returns:
        Accumulated result based on accumulate_fn
    """
    sensor = tel.sensors[sensor_idx]
    n_sources = sources.shape[0]

    stages = _get_stages(tel.optical_groups)
    stage_indices = sorted(stages.keys())

    if not stage_indices or 0 not in stages:
        # Return appropriate empty result based on accumulator
        dummy_pts = jnp.zeros((1, 2))
        dummy_sensor_idx = jnp.zeros((1,), dtype=jnp.int32)
        dummy_vals = jnp.zeros((1,))
        return accumulate_fn(dummy_pts, dummy_sensor_idx, dummy_vals, sensor, 1, 1) * 0

    data = _get_stage0_data(stages)
    n_elements = data['points'].shape[0]
    n_samples = data['points'].shape[1]

    def process_element(acc, element_idx):
        pts, s_idx, ray_vals = _trace_single_element(
            element_idx, data, sources, values, source_type,
            n_sources, n_samples, stage_indices, stages,
            tel.obstruction_groups, sensor
        )
        return acc + accumulate_fn(pts, s_idx, ray_vals, sensor, n_sources, n_samples), None

    # Initialize accumulator based on accumulate_fn
    # For _accumulate_image: (n_sensors, *sensor_shape)
    # For _accumulate_per_source: (n_sources, n_sensors * n_pixels)
    if accumulate_fn == _accumulate_image:
        acc_shape = (sensor.n_sensors,) + sensor.get_accumulator_shape()
        acc0 = jnp.zeros(acc_shape)
    else:
        n_pixels = math.prod(sensor.get_accumulator_shape())
        acc0 = jnp.zeros((n_sources, sensor.n_sensors * n_pixels))

    result, _ = jax.lax.scan(process_element, acc0, jnp.arange(n_elements))
    return result


def _render_debug_generic(tel, sources, values, source_type, sensor_idx):
    """Generic debug render that returns raw hits for all interaction types."""
    sensor = tel.sensors[sensor_idx]
    n_sources = sources.shape[0]

    stages = _get_stages(tel.optical_groups)
    stage_indices = sorted(stages.keys())

    if not stage_indices or 0 not in stages:
        return jnp.zeros((0, 2)), jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,))

    data = _get_stage0_data(stages)
    n_elements = data['points'].shape[0]
    n_samples = data['points'].shape[1]

    def process_element(carry, element_idx):
        pts, s_idx, ray_vals = _trace_single_element(
            element_idx, data, sources, values, source_type,
            n_sources, n_samples, stage_indices, stages,
            tel.obstruction_groups, sensor
        )
        return carry, (pts, s_idx, ray_vals)

    _, per_element = jax.lax.scan(process_element, None, jnp.arange(n_elements))
    # Concatenate results from all elements
    return (
        per_element[0].reshape(-1, 2),
        per_element[1].reshape(-1),
        per_element[2].reshape(-1)
    )


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render(tel, sources, values, source_type, sensor_idx=0):
    """Render sources through telescope onto sensor group.

    Supports mixed reflective/refractive optical systems. Stage 0 can be
    mirrors, lenses, or slabs. Stages 1+ can be any interaction type.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor group to use

    Returns:
        Accumulated image with shape (n_sensors, *per_sensor_shape).
        For square sensors: (n_sensors, height, width)
        For hexagonal sensors: (n_sensors, n_pixels)
    """
    return _render_generic(tel, sources, values, source_type, sensor_idx, _accumulate_image)


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_debug(tel, sources, values, source_type, sensor_idx=0):
    """Render without accumulation - returns raw hits.

    Supports mixed reflective/refractive optical systems.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor group to use

    Returns:
        Tuple of (points, sensor_idx, values) arrays with all ray intersections:
            - points: (n_rays, 2) sensor coordinates for each ray
            - sensor_idx: (n_rays,) index of sensor each ray hit
            - values: (n_rays,) final intensity values (0 if ray missed)
    """
    return _render_debug_generic(tel, sources, values, source_type, sensor_idx)


@partial(jax.jit, static_argnames=['source_type', 'sensor_idx'])
def render_response_matrix(tel, sources, values, source_type, sensor_idx=0):
    """Render multiple sources and return the source-to-pixel response matrix.

    This function traces N sources through the telescope and returns an N×M matrix
    where each row contains one source's contribution to all M pixels across all sensors.
    Uses incremental accumulation for memory efficiency.

    Supports mixed reflective/refractive optical systems.

    Args:
        tel: Telescope object
        sources: Source positions (n_sources, 3) for 'point' or directions for 'parallel'
        values: Source intensities (n_sources,)
        source_type: 'point' or 'parallel'
        sensor_idx: Index of sensor group to use

    Returns:
        Array of shape (n_sources, n_sensors * n_pixels) where n_pixels is the
        flattened per-sensor size.
    """
    return _render_generic(tel, sources, values, source_type, sensor_idx, _accumulate_per_source)


@partial(jax.jit, static_argnames=['sensor_idx'])
def trace_rays(tel, ray_origins, ray_directions, values, sensor_idx=0):
    """Render classical rays through telescope onto sensor group.

    Unlike render() which samples rays from primary mirror surfaces, this function
    traces rays from arbitrary external origins through the full optical system,
    including intersection with primary mirrors (stage 0).

    Supports mixed reflective/refractive systems - rays are automatically
    reflected or refracted based on each optical element's interaction type.

    Args:
        tel: Telescope object
        ray_origins: Ray starting positions (n_rays, 3)
        ray_directions: Ray directions (n_rays, 3), should be normalized
        values: Ray intensities (n_rays,)
        sensor_idx: Index of sensor group to use

    Returns:
        Accumulated image with shape (n_sensors, *per_sensor_shape).
    """
    sensor = tel.sensors[sensor_idx]

    stages = _get_stages(tel.optical_groups)
    stage_indices = sorted(stages.keys())

    if not stage_indices:
        acc_shape = (sensor.n_sensors,) + sensor.get_accumulator_shape()
        return jnp.zeros(acc_shape)

    pts, sensor_idx_arr, ray_vals = _trace_rays(
        ray_origins, ray_directions, values,
        stages, stage_indices, tel.obstruction_groups,
        sensor
    )

    return sensor.accumulate(sensor_idx_arr, pts[:, 0], pts[:, 1], ray_vals)


@partial(jax.jit, static_argnames=['sensor_idx'])
def trace_rays_debug(tel, ray_origins, ray_directions, values, sensor_idx=0):
    """Render classical rays without accumulation - returns raw hits.

    Debug version of trace_rays that returns individual ray hit positions
    and values instead of accumulated image.

    Supports mixed reflective/refractive systems.

    Args:
        tel: Telescope object
        ray_origins: Ray starting positions (n_rays, 3)
        ray_directions: Ray directions (n_rays, 3), should be normalized
        values: Ray intensities (n_rays,)
        sensor_idx: Index of sensor group to use

    Returns:
        Tuple of (points, sensor_idx, values):
            - points: (n_rays, 2) sensor coordinates for each ray
            - sensor_idx: (n_rays,) index of sensor each ray hit
            - values: (n_rays,) final intensity values (0 if ray missed)
    """
    sensor = tel.sensors[sensor_idx]

    stages = _get_stages(tel.optical_groups)
    stage_indices = sorted(stages.keys())

    if not stage_indices:
        return jnp.zeros((0, 2)), jnp.zeros((0,), dtype=jnp.int32), jnp.zeros((0,))

    pts, sensor_idx_arr, ray_vals = _trace_rays(
        ray_origins, ray_directions, values,
        stages, stage_indices, tel.obstruction_groups,
        sensor
    )

    return pts, sensor_idx_arr, ray_vals
