import jax
import jax.numpy as jnp
import jax.random as jr

from .ray_bundle import RayBundle
from .transforms import euler_to_matrix


def _get_stages(optical_groups):
    """Map optical_groups by optical_stage, return sorted dict (one group per stage)."""
    by_stage = {}
    for g in optical_groups:
        by_stage[g.optical_stage] = g
    return dict(sorted(by_stage.items()))


def _shadow_mask(origins, directions, obstructions, max_t=1e10):
    """Returns 1.0 where unoccluded, 0.0 where blocked."""
    if not obstructions:
        return jnp.ones(origins.shape[0])
    mask = jnp.ones(origins.shape[0])
    for g in obstructions:
        t = jax.vmap(g.intersect)(origins, directions)
        mask = mask * jnp.where(t < max_t, 0.0, 1.0)
    return mask


def _trace_stage(origins, directions, values, current_n, group, obstructions):
    """Process rays through one optical stage: intersect all elements, apply physics,
    check shadows.

    Returns ``(new_origins, new_directions, new_values, segment_length,
    opl_internal, new_n)``:

    * ``segment_length``: geometric distance from the previous stage
      to this surface (in the medium ``current_n``).
    * ``opl_internal``: per-ray OPL accumulated *inside* the
      interaction (non-zero only for slabs / windows).
    * ``new_n``: per-ray refractive index of the medium the ray is in
      after this stage, ready to weight the next segment.

    Uses lax.scan over elements, keeping only the closest hit per ray.
    Memory usage is O(n_rays) regardless of element count.
    """
    n_rays = origins.shape[0]

    def intersect_element(eidx):
        pos = group.positions[eidx]
        rot = euler_to_matrix(group.rotations[eidx])

        o_loc = jnp.einsum('ij,nj->ni', rot.T, origins - pos)
        d_loc = jnp.einsum('ij,nj->ni', rot.T, directions)

        ts, pts_loc, norms_loc = jax.vmap(
            lambda o, d: group.surface.intersect_at(eidx, o, d)
        )(o_loc, d_loc)

        aperture = group.check_aperture(pts_loc[:, 0], pts_loc[:, 1], eidx)
        ts = jnp.where(aperture, ts, jnp.inf)

        pts_w = jnp.einsum('ij,nj->ni', rot, pts_loc) + pos
        norms_w = jnp.einsum('ij,nj->ni', rot, norms_loc)
        return ts, pts_w, norms_w

    init_carry = (
        jnp.full(n_rays, jnp.inf),          # best_t
        jnp.zeros((n_rays, 3)),              # best_pts
        jnp.zeros((n_rays, 3)),              # best_norms
        jnp.zeros(n_rays, dtype=jnp.int32),  # best_elem
    )

    def scan_step(carry, eidx):
        best_t, best_pts, best_norms, best_elem = carry
        ts, pts_w, norms_w = intersect_element(eidx)
        closer = ts < best_t
        new_best_t = jnp.where(closer, ts, best_t)
        new_best_pts = jnp.where(closer[:, None], pts_w, best_pts)
        new_best_norms = jnp.where(closer[:, None], norms_w, best_norms)
        new_best_elem = jnp.where(closer, eidx.astype(jnp.int32), best_elem)
        return (new_best_t, new_best_pts, new_best_norms, new_best_elem), None

    (best_t, best_pts, best_norms, best_elem), _ = jax.lax.scan(
        scan_step, init_carry, jnp.arange(len(group))
    )

    # Apply surface roughness via BSDF module
    hit = best_t < 1e10
    safe_norms = jnp.where(hit[:, None], best_norms, jnp.array([0., 0., 1.]))
    roughness_key = jr.fold_in(group.sample_key, 0xB5DF01)
    best_norms = group.bsdf.perturb_normals(safe_norms, roughness_key, best_elem)

    new_dirs, new_origins, coeffs, opl_internal, new_n = (
        group.interaction_module.apply(
            directions, best_norms, best_pts, best_elem, current_n,
        )
    )

    shadow = _shadow_mask(origins, directions, obstructions, best_t)
    segment = jnp.where(hit, best_t, 0.0)
    opl_internal = jnp.where(hit, opl_internal, 0.0)
    # Rays that missed keep their medium; only rays that interacted update it.
    new_n = jnp.where(hit, new_n, current_n)
    return (
        new_origins, new_dirs, values * hit * shadow * coeffs,
        segment, opl_internal, new_n,
    )


def _build_primary_geometry(group):
    """Sample the primary element's aperture and apply surface roughness.

    Returns:
        Tuple of (points, normals, weights) arrays in world coordinates,
        each with shape (n_elements, n_samples, ...).
    """
    points, normals, weights = group.transform_to_world()
    roughness_key = jr.fold_in(group.sample_key, 0xB5DF00)
    normals = group.bsdf.perturb_normals(normals, roughness_key)
    return points, normals, weights


def _build_source_rays(points, normals, weights, sources, source_values,
                       source_type, obstruction_groups):
    """Generate flat ray arrays from sources aimed at one primary element.

    Args:
        points: (n_samples, 3) sampled surface points for one element.
        normals: (n_samples, 3) surface normals at those points.
        weights: (n_samples, 1) importance weights.
        sources: (n_sources, 3) source positions or unit propagation
            directions (depending on ``source_type``).
        source_values: (n_sources,) source intensities.
        source_type: 'point' or 'parallel'.
        obstruction_groups: list of ObstructionGroup for shadow testing.

    Returns:
        ``(origins, directions, normals, values, leg_in)`` all shaped
        (n_rays, ...). ``leg_in`` is the optical path length each ray
        already accumulated travelling from the source (or reference
        wavefront) to its primary sample point.
    """
    n_sources = sources.shape[0]
    n_samples = points.shape[0]
    n_rays = n_sources * n_samples

    if source_type == 'point':
        deltas = points[None, :, :] - sources[:, None, :]
        lengths = jnp.linalg.norm(deltas, axis=-1)
        dirs = deltas / lengths[..., None]
        # Distance from each source to each primary sample point.
        leg_in = lengths
    else:
        dirs = jnp.broadcast_to(sources[:, None, :], (n_sources, n_samples, 3))
        # Signed offset of each sample point from a reference wavefront
        # plane through the world origin, perpendicular to the source
        # direction. Per-source constant offset is absorbed; per-sample
        # *differences* exactly cancel the geometric sag of the primary.
        leg_in = (points[None, :, :] * dirs).sum(axis=-1)

    dirs_flat = dirs.reshape(n_rays, 3)
    origins_flat = jnp.broadcast_to(
        points[None, :, :], (n_sources, n_samples, 3)
    ).reshape(n_rays, 3)
    normals_flat = jnp.broadcast_to(
        normals[None, :, :], (n_sources, n_samples, 3)
    ).reshape(n_rays, 3)
    leg_in_flat = leg_in.reshape(n_rays)

    shadow = _shadow_mask(origins_flat, -dirs_flat, obstruction_groups)
    weights_flat = jnp.broadcast_to(
        weights[:, 0][None, :], (n_sources, n_samples)
    ).reshape(n_rays)
    vals = jnp.broadcast_to(
        source_values[:, None], (n_sources, n_samples)
    ).reshape(n_rays) / weights_flat * shadow

    return origins_flat, dirs_flat, normals_flat, vals, leg_in_flat


def _apply_primary_interaction(group, element_idx, origins, directions,
                               normals, values, current_n):
    """Apply stage-0 physics: interaction + cos-theta weighting.

    Returns:
        (new_origins, new_directions, updated_values, new_n).
    """
    n_rays = origins.shape[0]
    elem_indices = jnp.full((n_rays,), element_idx, dtype=jnp.int32)

    new_dirs, new_origins, coeffs, _opl_internal, new_n = (
        group.interaction_module.apply(
            directions, normals, origins, elem_indices, current_n,
        )
    )

    cos_theta = jnp.abs(jnp.sum(directions * normals, axis=-1))
    return new_origins, new_dirs, values * coeffs * cos_theta, new_n


def _empty_bundle() -> RayBundle:
    z = jnp.zeros(0)
    return RayBundle(
        origins=jnp.zeros((0, 3)), directions=jnp.zeros((0, 3)),
        values=z, path_length=z, n=z,
    )


def _trace_one_element(stages, stage_indices, geom, sources, values,
                       source_type, obstructions, eidx):
    """Trace rays from sources through one stage-0 element of the optics.

    Returns a per-element :class:`RayBundle` of length ``n_sources * n_samples``
    in world coordinates, source-major.
    """
    s0_points, s0_normals, s0_weights = geom
    origins, dirs, normals, vals, leg_in = _build_source_rays(
        s0_points[eidx], s0_normals[eidx], s0_weights[eidx],
        sources, values, source_type, obstructions,
    )
    current_n = jnp.ones(vals.shape[0])
    origins, dirs, vals, current_n = _apply_primary_interaction(
        stages[0], eidx, origins, dirs, normals, vals, current_n,
    )
    # Seed OPL with the source-to-primary leg so the Monte-Carlo sample
    # location on the primary doesn't conflate with downstream analysis.
    path_length = leg_in
    for sidx in stage_indices[1:]:
        origins, dirs, vals, seg, opl_internal, new_n = _trace_stage(
            origins, dirs, vals, current_n, stages[sidx], obstructions,
        )
        path_length = path_length + current_n * seg + opl_internal
        current_n = new_n
    return RayBundle(
        origins=origins, directions=dirs, values=vals,
        path_length=path_length, n=current_n,
    )


def _per_element_scan(optical_groups, obstructions, sources, values, source_type):
    """Common setup for both render variants.

    Returns ``(trace_one, n_elements)`` or ``None`` if the optics has
    no stage-0 group, where ``trace_one(eidx) -> RayBundle`` traces a
    single primary element.
    """
    stages = _get_stages(optical_groups)
    if 0 not in stages:
        return None
    stage_indices = sorted(stages.keys())
    geom = _build_primary_geometry(stages[0])
    n_elements = geom[0].shape[0]

    def trace_one(eidx):
        return _trace_one_element(
            stages, stage_indices, geom,
            sources, values, source_type, obstructions, eidx,
        )

    return trace_one, n_elements


def render_optics(optical_groups, obstruction_groups, sources, values, source_type):
    """Render sources through the optics; return one flat :class:`RayBundle`.

    Materialises the full ``(n_elements * n_sources * n_samples,)`` ray
    buffer. Use :func:`render_optics_accumulate` when only a small
    aggregate (image, response matrix, ...) is needed.
    """
    setup = _per_element_scan(
        optical_groups, obstruction_groups, sources, values, source_type,
    )
    if setup is None:
        return _empty_bundle()
    trace_one, n_elements = setup

    _, per_el = jax.lax.scan(
        lambda _c, e: (None, trace_one(e)), None, jnp.arange(n_elements),
    )
    return jax.tree_util.tree_map(
        lambda a: a.reshape((-1,) + a.shape[2:]), per_el,
    )


def render_optics_accumulate(
    optical_groups, obstruction_groups, sources, values, source_type,
    accumulator, init,
):
    """Carry-folding render: walk stage-0 elements with an accumulator.

    Calls ``accumulator(carry, per_element_bundle) -> carry`` for each
    primary element instead of stacking outputs. Peak memory is bounded
    by ``init`` plus one element's rays, regardless of element count.

    The per-element bundle has length ``n_sources * n_samples`` in world
    coordinates and is laid out source-major (the first ``n_samples``
    rays belong to ``sources[0]``).
    """
    setup = _per_element_scan(
        optical_groups, obstruction_groups, sources, values, source_type,
    )
    if setup is None:
        return init
    trace_one, n_elements = setup

    def step(carry, eidx):
        return accumulator(carry, trace_one(eidx)), None

    final, _ = jax.lax.scan(step, init, jnp.arange(n_elements))
    return final


def trace_optics(optical_groups, obstruction_groups, ray_origins, ray_directions, values):
    """Trace rays from arbitrary origins through full optical system.

    Args:
        optical_groups: List of OpticalElementGroup (combined mirrors + lenses).
        obstruction_groups: List of ObstructionGroup.
        ray_origins: (n_rays, 3).
        ray_directions: (n_rays, 3), normalized.
        values: (n_rays,).

    Returns:
        RayBundle with rays in 3D space after all optical stages.
    """
    stages = _get_stages(optical_groups)
    stage_indices = sorted(stages.keys())

    origins, dirs, vals = ray_origins, ray_directions, values
    path_length = jnp.zeros(vals.shape[0])
    current_n = jnp.ones(vals.shape[0])

    if stage_indices:
        for stage_idx in stage_indices:
            origins, dirs, vals, seg, opl_internal, new_n = _trace_stage(
                origins, dirs, vals, current_n, stages[stage_idx], obstruction_groups
            )
            path_length = path_length + current_n * seg + opl_internal
            current_n = new_n

    return RayBundle(
        origins=origins,
        directions=dirs,
        values=vals,
        path_length=path_length,
        n=current_n,
    )
