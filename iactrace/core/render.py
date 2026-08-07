import jax
import jax.numpy as jnp
from jax import Array

from .intersections import intersect_plane
from .ray_bundle import RayBundle
from .trajectory import TraceResult, Trajectory
from .transforms import euler_to_matrix

_PRIMARY_ROUGHNESS_SALT = 0xB5DF00
_ROUGHNESS_SALT = 0xB5DF01


def _get_stages(optical_groups):
    """Map optical_groups by optical_stage, return sorted dict (one group per stage)."""
    by_stage = {}
    for g in optical_groups:
        by_stage[g.optical_stage] = g
    return dict(sorted(by_stage.items()))


def _shadow_mask(origins, directions, obstructions, max_t):
    """Returns 1.0 where unoccluded, 0.0 where blocked."""
    if not obstructions:
        return jnp.ones(origins.shape[0])
    mask = jnp.ones(origins.shape[0])
    for g in obstructions:
        t = jax.vmap(g.intersect)(origins, directions)
        mask = mask * jnp.where(t < max_t, 0.0, 1.0)
    return mask


def apply_final_leg_shadow(rb, obstruction_groups, camera_position, camera_rotation):
    """Shadow the converging beam on the final last-optic -> focal-plane leg.

    ``rb`` must be a world-frame bundle as produced by the render, i.e.
    *before* :meth:`RayBundle.to_frame`: its ``origins`` lie on the last
    optic and its ``directions`` point toward the focal plane. Only
    ``values`` and ``alive`` is modified, so the leg's contribution to ``path_length``
    is still added later by the sensor intersection.

    The leg is capped at the **camera reference plane** (``camera_position``),
    not at each ray's true sensor / focal-surface landing point. For a thin
    camera (sensor ~ ``camera_position``) these coincide; if the sensor group
    or focal surface is offset along the optical axis they differ, and an
    obstruction between ``camera_position`` and the true landing plane is not
    accounted for.
    """
    if not obstruction_groups:
        return rb
    rot = euler_to_matrix(camera_rotation)
    _, t_cap = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
        rb.origins,
        rb.directions,
        camera_position,
        rot,
    )
    shadow = _shadow_mask(rb.origins, rb.directions, obstruction_groups, t_cap)
    new_alive = rb.alive & (shadow > 0)
    return RayBundle(
        origins=rb.origins,
        directions=rb.directions,
        values=jnp.where(new_alive, rb.values, 0.0),
        path_length=rb.path_length,
        n=rb.n,
        alive=new_alive,
    )


def final_leg_points(rb, camera_position, camera_rotation, fallback):
    """Where the final last-optic -> focal-plane leg lands, for trajectories.

    ``rb`` must be a world-frame bundle (origins on the last optic, directions
    pointing toward the focal plane), as produced by :func:`trace_optics` and
    shadowed by :func:`apply_final_leg_shadow`. Returns the ``(n_rays, 3)``
    intersection with the **camera reference plane** -- the same cap
    :func:`apply_final_leg_shadow` uses, so the drawn leg matches the shadowed
    one.

    Rays that are dead, or whose direction does not cross the plane ahead of
    them, keep ``fallback`` (their last valid position) instead of a
    meaningless extrapolation.
    """
    rot = euler_to_matrix(camera_rotation)
    _, t = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
        rb.origins,
        rb.directions,
        camera_position,
        rot,
    )
    reaches = rb.alive & jnp.isfinite(t) & (t > 0.0)
    landing = rb.origins + jnp.where(reaches, t, 0.0)[:, None] * rb.directions
    return jnp.where(reaches[:, None], landing, fallback)


def _trace_stage(
    origins,
    directions,
    values,
    alive,
    current_n,
    group,
    obstructions,
    roughness_salt=_ROUGHNESS_SALT,
):
    """Process rays through one optical stage: intersect all elements, apply physics,
    check shadows.

    Args:
        roughness_salt: Folded into the group's ``sample_key`` to draw this
            call's surface-roughness perturbation.

    Returns ``(new_origins, new_directions, new_values, new_alive,
    segment_length, opl_internal, new_n)``:

    * ``new_alive``: per-ray liveness after this stage. A ray dies here if
      it misses every element (or lands outside an aperture) or is blocked
      by an obstruction; the physical coefficients only attenuate a ray
      that is still alive.
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

    init_carry = (
        jnp.full(n_rays, jnp.inf),  # best_t
        jnp.zeros((n_rays, 3)),  # best_pts
        jnp.zeros((n_rays, 3)),  # best_norms
        jnp.zeros(n_rays, dtype=jnp.int32),  # best_elem
    )

    def scan_step(carry, eidx):
        best_t, best_pts, best_norms, best_elem = carry
        ts, pts_w, norms_w = group.intersect(eidx, origins, directions)
        closer = ts < best_t
        new_best_t = jnp.where(closer, ts, best_t)
        new_best_pts = jnp.where(closer[:, None], pts_w, best_pts)
        new_best_norms = jnp.where(closer[:, None], norms_w, best_norms)
        new_best_elem = jnp.where(closer, eidx.astype(jnp.int32), best_elem)
        return (new_best_t, new_best_pts, new_best_norms, new_best_elem), None

    (best_t, best_pts, best_norms, best_elem), _ = jax.lax.scan(
        scan_step, init_carry, jnp.arange(len(group))
    )

    # Guard against the degenerate init-carry normal for rays that hit nothing.
    hit = best_t < 1e10
    safe_norms = jnp.where(hit[:, None], best_norms, jnp.array([0.0, 0.0, 1.0]))

    new_dirs, new_origins, coeffs, opl_internal, new_n = group.interact(
        directions,
        safe_norms,
        best_pts,
        best_elem,
        current_n,
        roughness_salt=roughness_salt,
    )

    shadow = _shadow_mask(origins, directions, obstructions, best_t)
    new_alive = alive & hit & (shadow > 0)
    new_values = jnp.where(new_alive, values * coeffs, 0.0)
    segment = jnp.where(hit, best_t, 0.0)
    opl_internal = jnp.where(hit, opl_internal, 0.0)
    # Rays that missed keep their medium; only rays that interacted update it.
    new_n = jnp.where(hit, new_n, current_n)
    return (
        new_origins,
        new_dirs,
        new_values,
        new_alive,
        segment,
        opl_internal,
        new_n,
    )


def _build_primary_geometry(group):
    """Sample the primary element's aperture and apply surface roughness.

    Returns:
        Tuple of (points, normals, weights) arrays in world coordinates,
        each with shape (n_elements, n_samples, ...).
    """
    return group.sample_primary_geometry(roughness_salt=_PRIMARY_ROUGHNESS_SALT)


def _build_source_rays(
    points, normals, weights, sources, source_values, source_type, obstruction_groups
):
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
        ``(origins, directions, normals, values, alive, leg_in)`` all
        shaped (n_rays, ...). ``alive`` is ``False`` for rays whose
        source-to-primary segment is blocked by an obstruction. ``leg_in``
        is the optical path length each ray already accumulated travelling
        from the source (or reference wavefront) to its primary sample
        point.
    """
    n_sources = sources.shape[0]
    n_samples = points.shape[0]
    n_rays = n_sources * n_samples

    if source_type == "point":
        deltas = points[None, :, :] - sources[:, None, :]
        lengths = jnp.linalg.norm(deltas, axis=-1)
        dirs = deltas / lengths[..., None]
        leg_in = lengths
        irradiance = 1.0 / (lengths * lengths)
        shadow_cap = lengths
    else:
        units = sources / jnp.linalg.norm(sources, axis=-1, keepdims=True)
        dirs = jnp.broadcast_to(units[:, None, :], (n_sources, n_samples, 3))
        leg_in = (points[None, :, :] * dirs).sum(axis=-1)
        irradiance = jnp.ones((n_sources, n_samples), dtype=leg_in.dtype)
        shadow_cap = jnp.full((n_sources, n_samples), jnp.inf, dtype=leg_in.dtype)

    dirs_flat = dirs.reshape(n_rays, 3)
    origins_flat = jnp.broadcast_to(points[None, :, :], (n_sources, n_samples, 3)).reshape(
        n_rays, 3
    )
    normals_flat = jnp.broadcast_to(normals[None, :, :], (n_sources, n_samples, 3)).reshape(
        n_rays, 3
    )
    leg_in_flat = leg_in.reshape(n_rays)

    shadow = _shadow_mask(
        origins_flat,
        -dirs_flat,
        obstruction_groups,
        shadow_cap.reshape(n_rays),
    )
    # A blocked source-to-primary segment terminates the ray (geometry loss).
    alive = shadow > 0
    weights_flat = jnp.broadcast_to(weights[:, 0][None, :], (n_sources, n_samples)).reshape(n_rays)
    vals = jnp.broadcast_to(source_values[:, None], (n_sources, n_samples)).reshape(n_rays)
    vals = jnp.where(alive, vals * irradiance.reshape(n_rays) / weights_flat, 0.0)

    return origins_flat, dirs_flat, normals_flat, vals, alive, leg_in_flat


def _apply_primary_interaction(group, element_idx, origins, directions, normals, values, current_n):
    """Apply stage-0 physics: interaction + cos-theta weighting.

    Returns:
        (new_origins, new_directions, updated_values, opl_internal, new_n).
    """
    n_rays = origins.shape[0]
    elem_indices = jnp.full((n_rays,), element_idx, dtype=jnp.int32)

    new_dirs, new_origins, coeffs, opl_internal, new_n = group.apply_interaction(
        directions,
        normals,
        origins,
        elem_indices,
        current_n,
    )

    cos_theta = jnp.abs(jnp.sum(directions * normals, axis=-1))
    return new_origins, new_dirs, values * coeffs * cos_theta, opl_internal, new_n


def _empty_bundle() -> RayBundle:
    z = jnp.zeros(0)
    return RayBundle(
        origins=jnp.zeros((0, 3)),
        directions=jnp.zeros((0, 3)),
        values=z,
        path_length=z,
        n=z,
    )


def _trace_one_element(
    stages, stage_indices, geom, sources, values, source_type, obstructions, eidx
):
    """Trace rays from sources through one stage-0 element of the optics.

    Returns a per-element :class:`RayBundle` of length ``n_sources * n_samples``
    in world coordinates, source-major.
    """
    s0_points, s0_normals, s0_weights = geom
    origins, dirs, normals, vals, alive, leg_in = _build_source_rays(
        s0_points[eidx],
        s0_normals[eidx],
        s0_weights[eidx],
        sources,
        values,
        source_type,
        obstructions,
    )
    current_n = jnp.ones(vals.shape[0])
    origins, dirs, vals, opl_internal, current_n = _apply_primary_interaction(
        stages[0],
        eidx,
        origins,
        dirs,
        normals,
        vals,
        current_n,
    )
    # ``leg_in`` reaches the primary's front face; a stage-0 slab still adds
    # its own n * L on top before the ray leaves the element.
    path_length = leg_in + opl_internal
    for sidx in stage_indices[1:]:
        origins, dirs, vals, alive, seg, opl_internal, new_n = _trace_stage(
            origins,
            dirs,
            vals,
            alive,
            current_n,
            stages[sidx],
            obstructions,
            roughness_salt=_ROUGHNESS_SALT + eidx,
        )
        path_length = path_length + current_n * seg + opl_internal
        current_n = new_n
    return RayBundle(
        origins=origins,
        directions=dirs,
        values=vals,
        path_length=path_length,
        n=current_n,
        alive=alive,
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
            stages,
            stage_indices,
            geom,
            sources,
            values,
            source_type,
            obstructions,
            eidx,
        )

    return trace_one, n_elements


def render_optics(optical_groups, obstruction_groups, sources, values, source_type):
    """Render sources through the optics; return one flat :class:`RayBundle`.

    Materialises the full ``(n_elements * n_sources * n_samples,)`` ray
    buffer. Use :func:`render_optics_accumulate` when only a small
    aggregate (image, response matrix, ...) is needed.
    """
    setup = _per_element_scan(
        optical_groups,
        obstruction_groups,
        sources,
        values,
        source_type,
    )
    if setup is None:
        return _empty_bundle()
    trace_one, n_elements = setup

    _, per_el = jax.lax.scan(
        lambda _c, e: (None, trace_one(e)),
        None,
        jnp.arange(n_elements),
    )
    return jax.tree_util.tree_map(
        lambda a: a.reshape((-1,) + a.shape[2:]),
        per_el,
    )


def render_optics_accumulate(
    optical_groups,
    obstruction_groups,
    sources,
    values,
    source_type,
    accumulator,
    init,
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
        optical_groups,
        obstruction_groups,
        sources,
        values,
        source_type,
    )
    if setup is None:
        return init
    trace_one, n_elements = setup

    def step(carry, eidx):
        return accumulator(carry, trace_one(eidx)), None

    final, _ = jax.lax.scan(step, init, jnp.arange(n_elements))
    return final


def trace_optics(
    optical_groups,
    obstruction_groups,
    ray_origins,
    ray_directions,
    values,
    record_trajectory=False,
):
    """Trace rays from arbitrary origins through full optical system.

    Args:
        optical_groups: List of OpticalElementGroup (combined mirrors + lenses).
        obstruction_groups: List of ObstructionGroup.
        ray_origins: (n_rays, 3).
        ray_directions: (n_rays, 3), normalized.
        values: (n_rays,).
        record_trajectory: When True, also collect the per-stage hit points and
            return them as a :class:`~iactrace.core.trajectory.Trajectory`
            alongside the RayBundle. Off by default; when off, no trajectory is
            built and nothing extra is computed (mirrors the
            :func:`~iactrace.camera.trace_chain` ``record_trajectory`` option).

    Returns:
        A :class:`~iactrace.core.trajectory.TraceResult`. Its ``rays`` are in 3D
        space after all optical stages; its ``trajectory`` is ``None`` unless
        ``record_trajectory`` was set, in which case the
        :class:`~iactrace.core.trajectory.Trajectory` holds the source point
        followed by each stage's landing point (world frame),
        ``(n_stages + 1, n_rays, 3)``. It ends on the **last optic** -- this
        kernel knows no camera.
    """
    stages = _get_stages(optical_groups)
    stage_indices = sorted(stages.keys())

    origins, dirs, vals = ray_origins, ray_directions, values
    path_length = jnp.zeros(vals.shape[0])
    current_n = jnp.ones(vals.shape[0])
    alive = jnp.ones(vals.shape[0], dtype=bool)

    # First trajectory point is the source; each stage appends its landing point.
    trajectory: list[Array] | None = [ray_origins] if record_trajectory else None

    if stage_indices:
        for stage_idx in stage_indices:
            origins, dirs, vals, alive, seg, opl_internal, new_n = _trace_stage(
                origins, dirs, vals, alive, current_n, stages[stage_idx], obstruction_groups
            )
            path_length = path_length + current_n * seg + opl_internal
            current_n = new_n
            if trajectory is not None:
                trajectory.append(jnp.where(alive[:, None], origins, trajectory[-1]))

    rays = RayBundle(
        origins=origins,
        directions=dirs,
        values=vals,
        path_length=path_length,
        n=current_n,
        alive=alive,
    )
    if trajectory is None:
        return TraceResult(rays)
    return TraceResult(rays, Trajectory(points=jnp.stack(trajectory, axis=0)))
