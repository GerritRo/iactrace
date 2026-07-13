from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.interactions import ReflectInteraction, RefractInteraction, SlabInteraction
from ..core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)

if TYPE_CHECKING:
    from ..core import ObstructionGroup
    from ..core.apertures import Aperture
    from ..core.surfaces import SurfaceGroup
    from .telescope import Telescope


# Internal helpers


def _stage_to_list_and_idx(
    telescope: Telescope, stage: int
) -> tuple[Literal["mirror_groups", "lens_groups"], int]:
    """Locate which list a stage lives in and at what index."""
    for i, g in enumerate(telescope.mirror_groups):
        if g.optical_stage == stage:
            return ("mirror_groups", i)
    for i, g in enumerate(telescope.lens_groups):
        if g.optical_stage == stage:
            return ("lens_groups", i)
    available = sorted(g.optical_stage for g in telescope.optical_groups)
    raise IndexError(f"no stage {stage}; available: {available}")


def _update_at_stage(
    telescope: Telescope,
    stage: int,
    getter: Callable,
    new_value: Any,
) -> Telescope:
    """Update a single attribute on the group at ``stage``."""
    list_name, idx = _stage_to_list_and_idx(telescope, stage)
    groups = getattr(telescope, list_name)
    new_group = eqx.tree_at(getter, groups[idx], new_value)
    new_groups = list(groups)
    new_groups[idx] = new_group
    return eqx.tree_at(lambda t: getattr(t, list_name), telescope, new_groups)


def _require_kind(group, stage: int, kinds: tuple, expected: str) -> None:
    """Raise unless the stage's interaction is one of ``kinds``."""
    if not isinstance(group.interaction_module, kinds):
        raise ValueError(f"stage {stage} is {group.kind}; expected {expected}")


def _broadcast(value: Array | float, n: int) -> Array:
    """Coerce to a length-``n`` array: scalars fill, per-element arrays pass through."""
    arr = jnp.asarray(value)
    return jnp.full((n,), arr) if arr.ndim == 0 else arr


def _focal_scale(group, stage: int, n_outside: float = 1.0) -> Array | float:
    """Curvature<->focal-length scale: ``2`` for mirrors, ``n_in - n_out`` for lenses.

    ``n_outside`` is a design-time paraxial assumption, not a value read from
    the group: the traced interaction has no stored ambient index (the render
    loop resolves it dynamically from each ray's current medium), so callers
    pass the ambient index the desired focal length is defined against.
    Defaults to ``1.0`` (vacuum/air), matching the initial medium every ray
    starts in.

    Raises for slabs, where a focal length is not meaningful.
    """
    scale = group.interaction_module.focal_scale(n_outside)
    if scale is None:
        raise ValueError(f"stage {stage} is slab; focal length is not meaningful")
    return scale


# Surface capability dispatch


def _asphere_locator(surface: SurfaceGroup):
    """Return ``f: surface -> AsphericSurfaceGroup`` within it, or ``None``.

    The returned callable navigates from a surface to its aspheric component,
    so it doubles as an ``eqx.tree_at`` path: a bare asphere maps to itself, an
    asphere nested in a :class:`SumSurfaceGroup` maps to that component.
    """
    if isinstance(surface, AsphericSurfaceGroup):
        return lambda s: s
    if isinstance(surface, SumSurfaceGroup):
        for i, c in enumerate(surface.components):
            if isinstance(c, AsphericSurfaceGroup):
                return lambda s, i=i: s.components[i]
    return None


def _asphere_of(surface: SurfaceGroup) -> AsphericSurfaceGroup | None:
    """Return the :class:`AsphericSurfaceGroup` within ``surface``, or ``None``."""
    locate = _asphere_locator(surface)
    return None if locate is None else locate(surface)


def _require_asphere(surface: SurfaceGroup, stage: int) -> AsphericSurfaceGroup:
    """Return the aspheric component, or raise if the stage has none."""
    asph = _asphere_of(surface)
    if asph is None:
        raise ValueError(
            f"stage {stage} has no aspheric surface to modify (surface is {type(surface).__name__})"
        )
    return asph


def _update_surface_attr(
    telescope: Telescope,
    stage: int,
    attr_getter: Callable,
    new_value: Any,
) -> Telescope:
    """Write an attribute of the stage's aspheric component.

    Locates the :class:`AsphericSurfaceGroup` (bare or inside a
    :class:`SumSurfaceGroup`) and updates ``attr_getter(asphere)`` in place,
    so curvature / conic / aspheric edits keep working after a Zernike term has
    wrapped the surface in a sum.
    """
    group = telescope.stage(stage)
    locate = _asphere_locator(group.surface)
    if locate is None:
        raise ValueError(
            f"stage {stage} has no aspheric surface to modify "
            f"(surface is {type(group.surface).__name__})"
        )
    return _update_at_stage(
        telescope,
        stage,
        lambda g: attr_getter(locate(g.surface)),
        new_value,
    )


# Zernike figure-error composition


def _facet_radii(aperture: Aperture):
    """Per-element Zernike normalization radius derived from the aperture.

    Disk apertures use their outer radius; polygon apertures use the
    circumradius (farthest vertex). ``rho = 1`` at this radius.
    """
    from ..core.apertures import DiskAperture, PolygonAperture

    if isinstance(aperture, DiskAperture):
        return aperture.radii
    if isinstance(aperture, PolygonAperture):
        return jnp.max(jnp.linalg.norm(aperture.vertices, axis=-1), axis=-1)
    raise ValueError(
        f"cannot derive a Zernike normalization radius from "
        f"{type(aperture).__name__}; pass an explicit r_norm"
    )


def _zernike_of(surface: SurfaceGroup) -> ZernikeSurfaceGroup | None:
    """Return the :class:`ZernikeSurfaceGroup` within ``surface``, or ``None``."""
    if isinstance(surface, ZernikeSurfaceGroup):
        return surface
    if isinstance(surface, SumSurfaceGroup):
        for c in surface.components:
            if isinstance(c, ZernikeSurfaceGroup):
                return c
    return None


def _pad_coeffs(coeffs, width: int):
    """Pad ``(N, J)`` Zernike coefficients to ``(N, width)`` with zeros."""
    n, j = coeffs.shape
    if j >= width:
        return coeffs
    return jnp.concatenate([coeffs, jnp.zeros((n, width - j))], axis=1)


def _replace_zernike(surface: SurfaceGroup, new_zernike: ZernikeSurfaceGroup) -> SurfaceGroup:
    """Return ``surface`` with its Zernike term replaced by ``new_zernike``."""
    if isinstance(surface, ZernikeSurfaceGroup):
        return new_zernike
    if isinstance(surface, SumSurfaceGroup):
        comps = tuple(
            new_zernike if isinstance(c, ZernikeSurfaceGroup) else c for c in surface.components
        )
        return SumSurfaceGroup(comps, offsets=surface.offsets)
    raise ValueError("surface has no Zernike term to replace")


def _add_zernike_to_surface(surface: SurfaceGroup, added, r_norm) -> SurfaceGroup:
    """Return ``surface`` with ``added`` ``(N, J)`` Zernike coefficients added.

    If the surface already carries a :class:`ZernikeSurfaceGroup`, its
    coefficients are incremented (padded to a common width). Otherwise a new
    Zernike term is composed in: a bare surface is wrapped in a
    :class:`SumSurfaceGroup`, and an existing sum gains the term as a new
    component.
    """
    existing = _zernike_of(surface)
    if existing is not None:
        width = max(existing.coeffs.shape[1], added.shape[1])
        new_coeffs = _pad_coeffs(existing.coeffs, width) + _pad_coeffs(added, width)
        new_zernike = ZernikeSurfaceGroup(
            coeffs=new_coeffs,
            r_norm=existing.r_norm,
            offsets=existing.offsets,
        )
        return _replace_zernike(surface, new_zernike)

    new_zernike = ZernikeSurfaceGroup(coeffs=added, r_norm=r_norm)
    if isinstance(surface, SumSurfaceGroup):
        return SumSurfaceGroup([*surface.components, new_zernike], offsets=surface.offsets)
    return SumSurfaceGroup([surface, new_zernike])


# Generic operations (any kind)


def set_positions(telescope: Telescope, stage: int, positions: Array) -> Telescope:
    """Set element positions for the group at ``stage``."""
    return _update_at_stage(telescope, stage, lambda g: g.positions, jnp.asarray(positions))


def set_rotations(telescope: Telescope, stage: int, rotations: Array) -> Telescope:
    """Set element rotations (Euler XYZ degrees) for the group at ``stage``."""
    return _update_at_stage(telescope, stage, lambda g: g.rotations, jnp.asarray(rotations))


def apply_displacement(telescope: Telescope, stage: int, sigma_z: float, key: Array) -> Telescope:
    """Apply random Gaussian z-displacement to elements in the group at ``stage``."""
    group = telescope.stage(stage)
    delta_z = jax.random.normal(key, shape=(len(group),)) * sigma_z
    new_positions = group.positions.at[:, 2].add(delta_z)
    return _update_at_stage(telescope, stage, lambda g: g.positions, new_positions)


def apply_misalignment(
    telescope: Telescope,
    stage: int,
    sigma_h: float,
    sigma_v: float,
    key: Array,
) -> Telescope:
    """Apply random Gaussian tip/tilt to element orientations.

    ``sigma_h`` / ``sigma_v`` are in arcseconds.
    """
    group = telescope.stage(stage)
    n = len(group)
    sigma_h_deg = sigma_h / 3600.0
    sigma_v_deg = sigma_v / 3600.0
    key1, key2 = jax.random.split(key)
    delta_h = jax.random.normal(key1, shape=(n,)) * sigma_h_deg
    delta_v = jax.random.normal(key2, shape=(n,)) * sigma_v_deg
    new_rotations = group.rotations.at[:, 0].add(delta_v).at[:, 1].add(delta_h)
    return _update_at_stage(telescope, stage, lambda g: g.rotations, new_rotations)


def apply_roughness(telescope: Telescope, stage: int, sigma: float) -> Telescope:
    """Apply Gaussian BSDF roughness (arcseconds RMS) to the group at ``stage``."""
    from ..core.bsdf import GaussianBSDF

    n = len(telescope.stage(stage))
    new_bsdf = GaussianBSDF(scale=jnp.full(n, sigma))
    return _update_at_stage(telescope, stage, lambda g: g.bsdf, new_bsdf)


def set_curvatures(telescope: Telescope, stage: int, curvatures: Array) -> Telescope:
    """Set surface curvatures (1/R) for the group at ``stage``."""
    return _update_surface_attr(telescope, stage, lambda a: a.curvatures, jnp.asarray(curvatures))


def set_conics(telescope: Telescope, stage: int, conics: Array) -> Telescope:
    """Set surface conic constants for the group at ``stage``."""
    return _update_surface_attr(telescope, stage, lambda a: a.conics, jnp.asarray(conics))


def set_aspherics(telescope: Telescope, stage: int, aspherics: Array) -> Telescope:
    """Set surface aspheric coefficients for the group at ``stage``."""
    return _update_surface_attr(telescope, stage, lambda a: a.aspherics, jnp.asarray(aspherics))


def scale_curvatures(telescope: Telescope, stage: int, factor: Array | float) -> Telescope:
    """Multiply curvatures by ``factor`` (scalar or per-element)."""
    asph = _require_asphere(telescope.stage(stage).surface, stage)
    new = asph.curvatures * jnp.asarray(factor)
    return _update_surface_attr(telescope, stage, lambda a: a.curvatures, new)


def offset_curvatures(telescope: Telescope, stage: int, offset: Array | float) -> Telescope:
    """Add ``offset`` to curvatures (scalar or per-element)."""
    asph = _require_asphere(telescope.stage(stage).surface, stage)
    new = asph.curvatures + jnp.asarray(offset)
    return _update_surface_attr(telescope, stage, lambda a: a.curvatures, new)


def apply_conic_error(telescope: Telescope, stage: int, sigma: float, key: Array) -> Telescope:
    """Apply random Gaussian error to conic constants."""
    group = telescope.stage(stage)
    asph = _require_asphere(group.surface, stage)
    noise = jax.random.normal(key, shape=(len(group),))
    new = asph.conics + noise * sigma
    return _update_surface_attr(telescope, stage, lambda a: a.conics, new)


def apply_aspheric_error(telescope: Telescope, stage: int, sigmas: Array, key: Array) -> Telescope:
    """Apply random Gaussian errors to aspheric coefficients."""
    group = telescope.stage(stage)
    asph = _require_asphere(group.surface, stage)
    n = len(group)
    n_terms = asph.aspherics.shape[1]
    sigmas = jnp.asarray(sigmas)
    if sigmas.size < n_terms:
        sigmas = jnp.concatenate([sigmas, jnp.zeros(n_terms - sigmas.size)])
    else:
        sigmas = sigmas[:n_terms]
    noise = jax.random.normal(key, shape=(n, n_terms))
    new = asph.aspherics + noise * sigmas[None, :]
    return _update_surface_attr(telescope, stage, lambda a: a.aspherics, new)


def apply_zernike_error(
    telescope: Telescope,
    stage: int,
    sigmas: Array,
    key: Array,
) -> Telescope:
    """Add random Gaussian Zernike figure error to the surface at ``stage``.

    Draws independent per-element coefficients for each Noll mode and composes
    them onto the stage's surface: a bare aspheric mirror becomes an asphere +
    Zernike :class:`SumSurfaceGroup`, while a surface that already carries a
    Zernike term has the new draw *added* to it (matching :func:`apply_conic_error`
    and :func:`apply_aspheric_error`, which accumulate noise rather than replace).

    The Noll terms are RMS-normalized, so ``sigmas`` are RMS surface-error sigmas
    in metres. ``sigmas[m]`` scales Noll index ``m + 1``: index 0 is piston
    (removed by the surface re-zero), 1/2 tilt, 3 defocus, 4/5 astigmatism,
    6/7 coma, 8/9 trefoil, 10 spherical. Only the first 11 Noll terms are
    available; a longer ``sigmas`` raises.

    Args:
        stage: Optical stage to perturb.
        sigmas: Per-Noll-mode RMS sigmas in metres, shape ``(J,)`` with
            ``J <= 11``.
        key: JAX PRNG key for the coefficient draw.
    """
    group = telescope.stage(stage)
    n = len(group)
    sigmas = jnp.asarray(sigmas)
    noise = jax.random.normal(key, shape=(n, sigmas.shape[0])) * sigmas[None, :]
    r_norm = _facet_radii(group.aperture)
    new_surface = _add_zernike_to_surface(group.surface, noise, r_norm)
    return _update_at_stage(telescope, stage, lambda g: g.surface, new_surface)


def _named_aberration_sigmas(width: int, indices: tuple[int, ...], sigma: float):
    """Per-Noll-mode sigma vector with ``sigma`` at the given columns."""
    s = jnp.zeros(width)
    for i in indices:
        s = s.at[i].set(sigma)
    return s


def apply_astigmatism(telescope: Telescope, stage: int, sigma: float, key: Array) -> Telescope:
    """Add random astigmatism (Noll Z5/Z6) of RMS ``sigma`` metres per component."""
    return apply_zernike_error(telescope, stage, _named_aberration_sigmas(6, (4, 5), sigma), key)


def apply_coma(telescope: Telescope, stage: int, sigma: float, key: Array) -> Telescope:
    """Add random coma (Noll Z7/Z8) of RMS ``sigma`` metres per component."""
    return apply_zernike_error(telescope, stage, _named_aberration_sigmas(8, (6, 7), sigma), key)


def apply_trefoil(telescope: Telescope, stage: int, sigma: float, key: Array) -> Telescope:
    """Add random trefoil (Noll Z9/Z10) of RMS ``sigma`` metres per component."""
    return apply_zernike_error(telescope, stage, _named_aberration_sigmas(10, (8, 9), sigma), key)


def resample(telescope: Telescope, stage: int, key: Array) -> Telescope:
    """Refresh the Monte-Carlo sampling key on the group at ``stage``."""
    return _update_at_stage(telescope, stage, lambda g: g.sample_key, key)


# Kind-specific operations


def set_reflectivity(telescope: Telescope, stage: int, reflectivity: Array | float) -> Telescope:
    """Set per-element mirror reflectivity. Mirror stages only."""
    group = telescope.stage(stage)
    _require_kind(group, stage, (ReflectInteraction,), "mirror")
    assert isinstance(group.interaction_module, ReflectInteraction)
    r = _broadcast(reflectivity, len(group))
    new_interaction = group.interaction_module.with_reflectivity_scalar(r)
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def scale_reflectivity(telescope: Telescope, stage: int, factor: Array | float) -> Telescope:
    """Multiply mirror reflectivity by ``factor``. Mirror stages only.

    Scales the bulk multiplier ``reflectivity_scalar``; the coating on
    the interaction is left untouched.
    """
    group = telescope.stage(stage)
    _require_kind(group, stage, (ReflectInteraction,), "mirror")
    assert isinstance(group.interaction_module, ReflectInteraction)
    new_interaction = group.interaction_module.scaled_reflectivity(_broadcast(factor, len(group)))
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def set_transmittance(telescope: Telescope, stage: int, transmittance: Array | float) -> Telescope:
    """Set per-element bulk transmittance. Lens or slab stages only.

    Writes the bulk multiplier ``transmittance_scalar``; the coating
    on the interaction is left untouched.
    """
    group = telescope.stage(stage)
    _require_kind(group, stage, (RefractInteraction, SlabInteraction), "lens or slab")
    assert isinstance(group.interaction_module, (RefractInteraction, SlabInteraction))
    t = _broadcast(transmittance, len(group))
    new_interaction = group.interaction_module.with_transmittance_scalar(t)
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def scale_transmittance(telescope: Telescope, stage: int, factor: Array | float) -> Telescope:
    """Multiply bulk transmittance by ``factor``. Lens or slab stages only.

    Scales the bulk multiplier ``transmittance_scalar``; the coating on
    the interaction is left untouched.
    """
    group = telescope.stage(stage)
    _require_kind(group, stage, (RefractInteraction, SlabInteraction), "lens or slab")
    assert isinstance(group.interaction_module, (RefractInteraction, SlabInteraction))
    new_interaction = group.interaction_module.scaled_transmittance(_broadcast(factor, len(group)))
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def set_refractive_index(telescope: Telescope, stage: int, n_inside: Array | float) -> Telescope:
    """Set per-element refractive index. Lens or slab stages only."""
    group = telescope.stage(stage)
    _require_kind(group, stage, (RefractInteraction, SlabInteraction), "lens or slab")
    assert isinstance(group.interaction_module, (RefractInteraction, SlabInteraction))
    n = _broadcast(n_inside, len(group))
    new_interaction = group.interaction_module.with_n_inside(n)
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def set_thickness(telescope: Telescope, stage: int, thickness: Array | float) -> Telescope:
    """Set slab thickness in metres. Slab stages only."""
    group = telescope.stage(stage)
    _require_kind(group, stage, (SlabInteraction,), "slab")
    assert isinstance(group.interaction_module, SlabInteraction)
    t = _broadcast(thickness, len(group))
    new_interaction = group.interaction_module.with_thickness(t)
    return _update_at_stage(telescope, stage, lambda g: g.interaction_module, new_interaction)


def set_focal_lengths(
    telescope: Telescope, stage: int, focal_lengths: Array, n_outside: float = 1.0
) -> Telescope:
    """Set focal lengths via curvature.

    Mirror stages: ``c = 1 / (2 f)``.
    Lens stages (single refracting surface): ``c = 1 / ((n_inside - n_outside) f)``,
    where ``n_outside`` is a design-time ambient-index assumption (default
    ``1.0``), not a value stored on the lens.
    """
    group = telescope.stage(stage)
    f = jnp.asarray(focal_lengths)
    scale = _focal_scale(group, stage, n_outside)
    new_c = jnp.where(jnp.isinf(f), 0.0, 1.0 / (scale * f))
    return set_curvatures(telescope, stage, new_c)


def apply_focal_error(
    telescope: Telescope,
    stage: int,
    sigma: float,
    key: Array,
    relative: bool = False,
    n_outside: float = 1.0,
) -> Telescope:
    """Perturb focal lengths by Gaussian noise; update curvatures accordingly.

    Kind-aware: uses the mirror or single-refracting-lens formula. Slabs
    are rejected. ``n_outside`` is the design-time ambient index for the
    lens formula (default ``1.0``); see :func:`set_focal_lengths`.
    """
    group = telescope.stage(stage)
    curvatures = _require_asphere(group.surface, stage).curvatures
    safe = jnp.where(curvatures == 0, 1e-10, curvatures)

    scale = _focal_scale(group, stage, n_outside)
    f = 1.0 / (scale * safe)
    noise = jax.random.normal(key, shape=(len(group),))
    new_f = f * (1.0 + noise * sigma) if relative else f + noise * sigma
    new_c = jnp.where(curvatures == 0, 0.0, 1.0 / (scale * new_f))

    return _update_surface_attr(telescope, stage, lambda a: a.curvatures, new_c)


# Obstruction operations


def add_obstruction(telescope: Telescope, obstruction: ObstructionGroup) -> Telescope:
    """Append an obstruction group."""
    new_groups = list(telescope.obstruction_groups) + [obstruction]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def remove_obstruction(telescope: Telescope, group_idx: int) -> Telescope:
    """Remove the obstruction group at ``group_idx``."""
    if not telescope.obstruction_groups:
        raise IndexError("No obstruction groups to remove")
    if group_idx < 0 or group_idx >= len(telescope.obstruction_groups):
        raise IndexError(
            f"Obstruction group index {group_idx} out of range "
            f"(0-{len(telescope.obstruction_groups) - 1})"
        )
    new_groups = [g for i, g in enumerate(telescope.obstruction_groups) if i != group_idx]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def clear_obstructions(telescope: Telescope) -> Telescope:
    """Drop all obstruction groups."""
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, [])


def get_obstruction_count(telescope: Telescope) -> int:
    """Total number of obstruction elements across all groups."""
    return sum(len(g) for g in telescope.obstruction_groups)


# Summary


def get_info(telescope: Telescope) -> dict[str, Any]:
    """Summary dict of telescope configuration."""
    from ..core.apertures import DiskAperture, PolygonAperture

    aperture_kind = {DiskAperture: "disk", PolygonAperture: "polygon"}

    stages_info = []
    for s in telescope.stage_indices():
        g = telescope.stage(s)
        ap = aperture_kind.get(type(g.aperture), "unknown")
        stages_info.append({"stage": s, "kind": g.kind, "n_elements": g.n_elements, "aperture": ap})

    if telescope.optical_groups:
        all_positions = jnp.concatenate([g.positions for g in telescope.optical_groups], axis=0)
        bbox_min = all_positions.min(axis=0)
        bbox_max = all_positions.max(axis=0)
    else:
        bbox_min = bbox_max = jnp.zeros(3)

    return {
        "name": telescope.name,
        "n_stages": telescope.n_stages,
        "stages": stages_info,
        "n_mirror_elements": telescope.n_mirror_elements,
        "n_lens_elements": telescope.n_lens_elements,
        "n_obstruction_groups": len(telescope.obstruction_groups),
        "n_obstructions": get_obstruction_count(telescope),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }
