from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from . import _salts
from .intersections import intersect_plane, is_hit
from .ray_bundle import DEFAULT_WAVELENGTH, RayBundle
from .spectrum import Spectrum, as_spectrum
from .trajectory import TraceResult, Trajectory
from .transforms import euler_to_matrix

# Ray seeding and source generation


def _seed_wavelengths(n_rays, wavelength, key):
    """Per-ray (n_rays,) wavelengths for a caller-supplied bundle.

    A trace is handed its rays, so a scalar (or None) means every ray at
    that wavelength, a (n_rays,) array is used as given, and a
    Spectrum draws one wavelength per ray.
    """
    if isinstance(wavelength, Spectrum):
        return wavelength.sample(key, (n_rays,))
    wl = jnp.asarray(DEFAULT_WAVELENGTH if wavelength is None else wavelength)
    if wl.ndim > 1 or (wl.ndim == 1 and wl.shape[0] != n_rays):
        raise ValueError(
            f"wavelength must be a scalar or one value per ray ({n_rays},), got "
            f"shape {wl.shape}. `trace` takes the rays you give it, so an array "
            "is per-ray; pass a Spectrum to draw one wavelength per ray."
        )
    return jnp.broadcast_to(wl, (n_rays,))


def _build_source_rays(
    points,
    normals,
    weights,
    sources,
    source_values,
    source_type,
    obstruction_groups,
    spectrum,
    wl_key,
):
    """Generate flat ray arrays from sources aimed at one primary element.

    Parameters
    ----------
    points : array, shape (n_samples, 3)
        Sampled surface points for one element.
    normals : array, shape (n_samples, 3)
        Surface normals at those points.
    weights : array, shape (n_samples, 1)
        Importance weights.
    sources : array, shape (n_sources, 3)
        Source positions or unit propagation directions (depending on source_type).
    source_values : array, shape (n_sources,)
        Source intensities.
    source_type
        'point' or 'parallel'.
    obstruction_groups
        list of ObstructionGroup for shadow testing.
    spectrum
        The source's Spectrum.
    wl_key
        PRNG key for that draw.

    Returns
    -------
    (origins, directions, normals, values, alive, leg_in,
    wavelength) all shaped (n_rays, ...), source-major.
    alive is False for rays whose source-to-primary segment is
    blocked by an obstruction. leg_in is the optical path length each
    ray already accumulated travelling from the source (or reference
    wavefront) to its primary sample point.
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

    return (
        origins_flat,
        dirs_flat,
        normals_flat,
        vals,
        alive,
        leg_in_flat,
        spectrum.sample(wl_key, (n_rays,)),
    )


def _apply_primary_interaction(
    group, element_idx, origins, directions, normals, values, current_n, wavelength
):
    """Apply stage-0 physics: interaction + cos-theta weighting.

    Returns
    -------
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
        wavelength,
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


# Stage kernel


def _shadow_mask(origins, directions, obstructions, max_t):
    """Returns 1.0 where unoccluded, 0.0 where blocked."""
    if not obstructions:
        return jnp.ones(origins.shape[0])
    mask = jnp.ones(origins.shape[0])
    for g in obstructions:
        t = g.intersect_batch(origins, directions)
        mask = mask * jnp.where(t < max_t, 0.0, 1.0)
    return mask


def _get_stages(optical_groups):
    """(stages, stage_indices): one group per optical stage, in stage order."""
    by_stage = {}
    for g in optical_groups:
        by_stage[g.optical_stage] = g
    stages = dict(sorted(by_stage.items()))
    return stages, list(stages)


def _nearest_hit(group, origins, directions):
    """Nearest element each ray hits in group: (t, element index).

    Every ray is tested against every element, and the
    smallest forward t wins.
    """
    n_rays = origins.shape[0]
    init_carry = (
        jnp.full(n_rays, jnp.inf, dtype=origins.dtype),  # best_t
        jnp.zeros(n_rays, dtype=jnp.int32),  # best_elem
    )

    def scan_step(carry, eidx):
        best_t, best_elem = carry
        ts = group.intersect_t(eidx, origins, directions)
        closer = ts < best_t
        return (
            jnp.where(closer, ts, best_t),
            jnp.where(closer, eidx.astype(jnp.int32), best_elem),
        ), None

    (best_t, best_elem), _ = jax.lax.scan(scan_step, init_carry, jnp.arange(len(group)))
    return best_t, best_elem


def _trace_stage(
    origins,
    directions,
    values,
    alive,
    current_n,
    wavelength,
    group,
    obstructions,
    roughness_salt=_salts.ROUGHNESS,
):
    """Process rays through one optical stage: intersect all elements, apply physics,
    check shadows.

    Parameters
    ----------
    roughness_salt
        Folded into the group's sample_key to draw this call's
        surface-roughness perturbation.

    Returns
    -------
    tuple
        (new_origins, new_directions, new_values, new_alive, segment_length,
        opl_internal, new_n), where:

        - new_alive is per-ray liveness after this stage. A ray dies here if
          it misses every element (or lands outside an aperture) or is
          blocked by an obstruction; the physical coefficients only attenuate
          a ray that is still alive.
        - segment_length is the geometric distance from the previous stage to
          this surface (in the medium current_n).
        - opl_internal is the per-ray OPL accumulated inside the interaction
          (non-zero only for slabs / windows).
        - new_n is the per-ray refractive index of the medium the ray is in
          after this stage, ready to weight the next segment.
    """
    best_t, best_elem = _nearest_hit(group, origins, directions)
    best_pts, best_norms = group.hit_geometry(best_elem, origins, directions)

    # Rays that hit nothing kept element 0's geometry
    hit = is_hit(best_t)
    safe_norms = jnp.where(hit[:, None], best_norms, jnp.array([0.0, 0.0, 1.0]))

    new_dirs, new_origins, coeffs, opl_internal, new_n = group.interact(
        directions,
        safe_norms,
        best_pts,
        best_elem,
        current_n,
        roughness_salt=roughness_salt,
        wavelength=wavelength,
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


def _trace_one_element(
    stages, stage_indices, geom, sources, values, source_type, obstructions, spectrum, eidx
):
    """Trace rays from sources through one stage-0 element of the optics.

    Returns a per-element RayBundle of length n_sources *
    n_samples in world coordinates, source-major. Wavelengths are drawn per
    ray from spectrum.
    """
    s0_points, s0_normals, s0_weights = geom
    origins, dirs, normals, vals, alive, leg_in, wl = _build_source_rays(
        s0_points[eidx],
        s0_normals[eidx],
        s0_weights[eidx],
        sources,
        values,
        source_type,
        obstructions,
        spectrum,
        jax.random.fold_in(stages[0].sample_key, _salts.WAVELENGTH + eidx),
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
        wl,
    )
    # leg_in reaches the primary's front face; a stage-0 slab still adds
    # its own n * L on top before the ray leaves the element.
    path_length = leg_in + opl_internal
    for sidx in stage_indices[1:]:
        origins, dirs, vals, alive, seg, opl_internal, new_n = _trace_stage(
            origins,
            dirs,
            vals,
            alive,
            current_n,
            wl,
            stages[sidx],
            obstructions,
            roughness_salt=_salts.ROUGHNESS + eidx,
        )
        path_length = path_length + current_n * seg + opl_internal
        current_n = new_n
    return RayBundle(
        origins=origins,
        directions=dirs,
        values=vals,
        path_length=path_length,
        n=current_n,
        wavelength=wl,
        alive=alive,
    )


def _per_element_scan(optical_groups, obstructions, sources, values, source_type, spectrum):
    """Common setup for both render variants.

    Returns (trace_one, n_elements) or None if the optics has
    no stage-0 group, where trace_one(eidx) -> RayBundle traces a
    single primary element. spectrum may be None for the
    monochromatic default.
    """
    if spectrum is None:
        spectrum = as_spectrum(DEFAULT_WAVELENGTH)
    stages, stage_indices = _get_stages(optical_groups)
    if 0 not in stages:
        return None
    geom = stages[0].sample_primary_geometry(roughness_salt=_salts.PRIMARY_ROUGHNESS)
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
            spectrum,
            eidx,
        )

    return trace_one, n_elements


# Public entrypoints


def render_optics(
    optical_groups,
    obstruction_groups,
    sources,
    values,
    source_type,
    *,
    spectrum=None,
):
    """Render sources through the optics; return one flat RayBundle.

    Materialises the full (n_elements * n_sources * n_samples,) ray
    buffer. Use render_optics_accumulate when only a small
    aggregate (image, response matrix, ...) is needed.

    spectrum is the source's
    Spectrum (default: monochromatic at
    DEFAULT_WAVELENGTH).
    """
    setup = _per_element_scan(
        optical_groups, obstruction_groups, sources, values, source_type, spectrum
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
    *,
    spectrum=None,
):
    """Carry-folding render: walk stage-0 elements with an accumulator.

    Calls accumulator(carry, per_element_bundle) -> carry for each
    primary element instead of stacking outputs. Peak memory is bounded
    by init plus one element's rays, regardless of element count.

    The per-element bundle has length n_sources * n_samples in world
    coordinates, source-major (the first n_samples rays belong to
    sources[0]). spectrum is the source's
    Spectrum (default: monochromatic at
    DEFAULT_WAVELENGTH).
    """
    setup = _per_element_scan(
        optical_groups, obstruction_groups, sources, values, source_type, spectrum
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
    *,
    wavelength=None,
):
    """Trace rays from arbitrary origins through full optical system.

    Parameters
    ----------
    optical_groups
        List of OpticalElementGroup (combined mirrors + lenses).
    obstruction_groups
        List of ObstructionGroup.
    ray_origins : array, shape (n_rays, 3)
        .
    ray_directions : array, shape (n_rays, 3)
        Normalized.
    values : array, shape (n_rays,)
        .
    wavelength : array, shape (n_rays,)
        a scalar shared by every ray, a per-ray.
        array, or a Spectrum, which draws
        one wavelength per ray. Default DEFAULT_WAVELENGTH.
    record_trajectory
        When True, also collect the per-stage hit points and
        return them as a Trajectory
        alongside the RayBundle.

    Returns
    -------
    A TraceResult. Its rays are in 3D
    space after all optical stages; its trajectory is None unless
    record_trajectory was set, in which case the
    Trajectory holds the source point
    followed by each stage's landing point (world frame),
    (n_stages + 1, n_rays, 3). It ends on the last optic.
    """
    stages, stage_indices = _get_stages(optical_groups)

    n_rays = values.shape[0]
    origins, dirs, vals = ray_origins, ray_directions, values
    wl = _seed_wavelengths(
        n_rays,
        wavelength,
        jax.random.fold_in(stages[stage_indices[0]].sample_key, _salts.WAVELENGTH),
    )
    path_length = jnp.zeros(n_rays)
    current_n = jnp.ones(n_rays)
    alive = jnp.ones(n_rays, dtype=bool)

    # First trajectory point is the source; each stage appends its landing point.
    trajectory: list[Array] | None = [origins] if record_trajectory else None

    for stage_idx in stage_indices:
        origins, dirs, vals, alive, seg, opl_internal, new_n = _trace_stage(
            origins, dirs, vals, alive, current_n, wl, stages[stage_idx], obstruction_groups
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
        wavelength=wl,
        alive=alive,
    )
    if trajectory is None:
        return TraceResult(rays)
    return TraceResult(rays, Trajectory(points=jnp.stack(trajectory, axis=0)))


class LazyRayBundle(eqx.Module):
    """A RayBundle described by a render that has not been evaluated yet.

    Holds the optics, obstructions, camera frame, and source
    description needed to evaluate itself. Output is delivered in the local
    frame defined by camera_position and camera_rotation.

    Consume it either with fold -- walk per primary-mirror element with
    an accumulator, so the full (n_elements * n_sources * n_samples,) ray
    buffer is never materialised -- or with materialise, when the
    per-ray output itself is the result (spot diagrams, Camera.collect).
    """

    optical_groups: list
    obstruction_groups: list
    camera_position: Array
    camera_rotation: Array
    sources: Array
    source_values: Array
    spectrum: Spectrum
    source_type: Literal["point", "parallel"] = eqx.field(static=True)

    def fold(self, accumulator, init):
        """Per-element scan: accumulator(carry, rb_local) -> carry.

        rb_local is one element's RayBundle, already handed off
        into the local frame.
        """
        origin, rotation = self.camera_position, self.camera_rotation
        obstructions = self.obstruction_groups

        def in_local_frame(carry, rb_world):
            return accumulator(carry, handoff_to_frame(rb_world, obstructions, origin, rotation))

        return render_optics_accumulate(
            self.optical_groups,
            self.obstruction_groups,
            self.sources,
            self.source_values,
            self.source_type,
            in_local_frame,
            init,
            spectrum=self.spectrum,
        )

    def materialise(self) -> RayBundle:
        """Run the render eagerly; return a flat local-frame RayBundle."""
        rb_world = render_optics(
            self.optical_groups,
            self.obstruction_groups,
            self.sources,
            self.source_values,
            self.source_type,
            spectrum=self.spectrum,
        )
        return handoff_to_frame(
            rb_world,
            self.obstruction_groups,
            self.camera_position,
            self.camera_rotation,
        )


# Handoff to a local frame


def _cap_at_plane(rb, plane_position, plane_rotation):
    """Ray parameter at which each ray crosses the given plane."""
    rot = euler_to_matrix(plane_rotation)
    _, t = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
        rb.origins, rb.directions, plane_position, rot
    )
    return t


def apply_final_leg_shadow(rb, obstruction_groups, plane_position, plane_rotation):
    """Shadow the converging beam on the final last-optic -> focal-plane leg."""
    if not obstruction_groups:
        return rb
    t_cap = _cap_at_plane(rb, plane_position, plane_rotation)
    shadow = _shadow_mask(rb.origins, rb.directions, obstruction_groups, t_cap)
    new_alive = rb.alive & (shadow > 0)
    return rb.replace(values=jnp.where(new_alive, rb.values, 0.0), alive=new_alive)


def final_leg_points(rb, plane_position, plane_rotation, fallback):
    """Where the final last-optic -> focal-plane leg lands."""
    t = _cap_at_plane(rb, plane_position, plane_rotation)
    reaches = rb.alive & jnp.isfinite(t) & (t > 0.0)
    landing = rb.origins + jnp.where(reaches, t, 0.0)[:, None] * rb.directions
    return jnp.where(reaches[:, None], landing, fallback)


def handoff_to_frame(rb, obstruction_groups, position, rotation):
    """Hand a world-frame bundle off to a local frame: shadow, then reframe."""
    return apply_final_leg_shadow(rb, obstruction_groups, position, rotation).to_frame(
        position, rotation
    )
