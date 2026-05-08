"""Functional operations on a :class:`~iactrace.Telescope`.

All operations are addressed by ``stage`` — the integer ``optical_stage``
of the target :class:`OpticalElementGroup`. The split between
``mirror_groups`` and ``lens_groups`` on the Telescope is purely a
storage / YAML detail; users only ever talk in stages.

Generic operations work on any kind of stage (mirror, lens, or slab).
Kind-specific operations validate at runtime and raise ``ValueError``
when called on the wrong kind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from ..core import ObstructionGroup
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


def _require_kind(
    telescope: Telescope, stage: int, allowed: set[str]
) -> str:
    """Validate the kind of the group at ``stage`` and return it."""
    kind = telescope.stage(stage).kind
    if kind not in allowed:
        raise ValueError(
            f"stage {stage} is {kind}; expected one of {sorted(allowed)}"
        )
    return kind


# Generic operations (any kind)


def set_positions(telescope: Telescope, stage: int, positions: Array) -> Telescope:
    """Set element positions for the group at ``stage``."""
    return _update_at_stage(
        telescope, stage, lambda g: g.positions, jnp.asarray(positions)
    )


def set_rotations(telescope: Telescope, stage: int, rotations: Array) -> Telescope:
    """Set element rotations (Euler XYZ degrees) for the group at ``stage``."""
    return _update_at_stage(
        telescope, stage, lambda g: g.rotations, jnp.asarray(rotations)
    )


def apply_displacement(
    telescope: Telescope, stage: int, sigma_z: float, key: Array
) -> Telescope:
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
    return _update_at_stage(
        telescope, stage, lambda g: g.surface.curvatures, jnp.asarray(curvatures)
    )


def set_conics(telescope: Telescope, stage: int, conics: Array) -> Telescope:
    """Set surface conic constants for the group at ``stage``."""
    return _update_at_stage(
        telescope, stage, lambda g: g.surface.conics, jnp.asarray(conics)
    )


def set_aspherics(telescope: Telescope, stage: int, aspherics: Array) -> Telescope:
    """Set surface aspheric coefficients for the group at ``stage``."""
    return _update_at_stage(
        telescope, stage, lambda g: g.surface.aspherics, jnp.asarray(aspherics)
    )


def scale_curvatures(
    telescope: Telescope, stage: int, factor: Array | float
) -> Telescope:
    """Multiply curvatures by ``factor`` (scalar or per-element)."""
    new = telescope.stage(stage).surface.curvatures * jnp.asarray(factor)
    return _update_at_stage(telescope, stage, lambda g: g.surface.curvatures, new)


def offset_curvatures(
    telescope: Telescope, stage: int, offset: Array | float
) -> Telescope:
    """Add ``offset`` to curvatures (scalar or per-element)."""
    new = telescope.stage(stage).surface.curvatures + jnp.asarray(offset)
    return _update_at_stage(telescope, stage, lambda g: g.surface.curvatures, new)


def apply_conic_error(
    telescope: Telescope, stage: int, sigma: float, key: Array
) -> Telescope:
    """Apply random Gaussian error to conic constants."""
    group = telescope.stage(stage)
    noise = jax.random.normal(key, shape=(len(group),))
    new = group.surface.conics + noise * sigma
    return _update_at_stage(telescope, stage, lambda g: g.surface.conics, new)


def apply_aspheric_error(
    telescope: Telescope, stage: int, sigmas: Array, key: Array
) -> Telescope:
    """Apply random Gaussian errors to aspheric coefficients."""
    group = telescope.stage(stage)
    n = len(group)
    n_terms = group.surface.aspherics.shape[1]
    sigmas = jnp.asarray(sigmas)
    if sigmas.size < n_terms:
        sigmas = jnp.concatenate([sigmas, jnp.zeros(n_terms - sigmas.size)])
    else:
        sigmas = sigmas[:n_terms]
    noise = jax.random.normal(key, shape=(n, n_terms))
    new = group.surface.aspherics + noise * sigmas[None, :]
    return _update_at_stage(telescope, stage, lambda g: g.surface.aspherics, new)


def resample(telescope: Telescope, stage: int, key: Array) -> Telescope:
    """Refresh the Monte-Carlo sampling key on the group at ``stage``."""
    return _update_at_stage(telescope, stage, lambda g: g.sample_key, key)


# Kind-specific operations


def set_reflectivity(
    telescope: Telescope, stage: int, reflectivity: Array | float
) -> Telescope:
    """Set per-element mirror reflectivity. Mirror stages only."""
    _require_kind(telescope, stage, {"mirror"})
    n = len(telescope.stage(stage))
    r = jnp.asarray(reflectivity)
    if r.ndim == 0:
        r = jnp.full(n, r)
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.reflectivity, r
    )


def scale_reflectivity(
    telescope: Telescope, stage: int, factor: Array | float
) -> Telescope:
    """Multiply mirror reflectivity by ``factor``. Mirror stages only."""
    _require_kind(telescope, stage, {"mirror"})
    group = telescope.stage(stage)
    factor = jnp.asarray(factor)
    if factor.ndim == 0:
        factor = jnp.full(len(group), factor)
    new = group.interaction_module.reflectivity * factor
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.reflectivity, new
    )


def set_transmittance(
    telescope: Telescope, stage: int, transmittance: Array | float
) -> Telescope:
    """Set per-element bulk transmittance. Lens or slab stages only."""
    _require_kind(telescope, stage, {"lens", "slab"})
    n = len(telescope.stage(stage))
    t = jnp.asarray(transmittance)
    if t.ndim == 0:
        t = jnp.full(n, t)
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.transmittance,
        jnp.clip(t, 0.0, 1.0),
    )


def scale_transmittance(
    telescope: Telescope, stage: int, factor: Array | float
) -> Telescope:
    """Multiply bulk transmittance by ``factor``. Lens or slab stages only."""
    _require_kind(telescope, stage, {"lens", "slab"})
    group = telescope.stage(stage)
    factor = jnp.asarray(factor)
    if factor.ndim == 0:
        factor = jnp.full(len(group), factor)
    new = jnp.clip(group.interaction_module.transmittance * factor, 0.0, 1.0)
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.transmittance, new
    )


def set_refractive_index(
    telescope: Telescope, stage: int, n_inside: Array | float
) -> Telescope:
    """Set per-element refractive index. Lens or slab stages only."""
    _require_kind(telescope, stage, {"lens", "slab"})
    n_elements = len(telescope.stage(stage))
    n = jnp.asarray(n_inside)
    if n.ndim == 0:
        n = jnp.full(n_elements, n)
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.n_inside, n
    )


def set_thickness(
    telescope: Telescope, stage: int, thickness: Array | float
) -> Telescope:
    """Set slab thickness in metres. Slab stages only."""
    _require_kind(telescope, stage, {"slab"})
    n = len(telescope.stage(stage))
    t = jnp.asarray(thickness)
    if t.ndim == 0:
        t = jnp.full(n, t)
    return _update_at_stage(
        telescope, stage, lambda g: g.interaction_module.thickness, t
    )


def set_focal_lengths(
    telescope: Telescope, stage: int, focal_lengths: Array
) -> Telescope:
    """Set focal lengths via curvature.

    Mirror stages: ``c = 1 / (2 f)``.
    Lens stages (single refracting surface): ``c = 1 / ((n - 1) f)``.
    Slab stages: not meaningful — raises ``ValueError``.
    """
    kind = _require_kind(telescope, stage, {"mirror", "lens"})
    f = jnp.asarray(focal_lengths)
    if kind == "mirror":
        new_c = jnp.where(jnp.isinf(f), 0.0, 1.0 / (2.0 * f))
    else:  # lens
        group = telescope.stage(stage)
        n_inside = group.interaction_module.n_inside
        n_outside = group.interaction_module.n_outside
        delta_n = n_inside - n_outside
        new_c = jnp.where(jnp.isinf(f), 0.0, 1.0 / (delta_n * f))
    return set_curvatures(telescope, stage, new_c)


def apply_focal_error(
    telescope: Telescope,
    stage: int,
    sigma: float,
    key: Array,
    relative: bool = False,
) -> Telescope:
    """Perturb focal lengths by Gaussian noise; update curvatures accordingly.

    Kind-aware: uses the mirror or single-refracting-lens formula. Slabs
    are rejected.
    """
    kind = _require_kind(telescope, stage, {"mirror", "lens"})
    group = telescope.stage(stage)
    curvatures = group.surface.curvatures
    safe = jnp.where(curvatures == 0, 1e-10, curvatures)

    if kind == "mirror":
        f = 1.0 / (2.0 * safe)
    else:  # lens
        delta_n = group.interaction_module.n_inside - group.interaction_module.n_outside
        f = 1.0 / (delta_n * safe)

    noise = jax.random.normal(key, shape=(len(group),))
    new_f = f * (1.0 + noise * sigma) if relative else f + noise * sigma

    if kind == "mirror":
        new_c = jnp.where(curvatures == 0, 0.0, 1.0 / (2.0 * new_f))
    else:
        delta_n = group.interaction_module.n_inside - group.interaction_module.n_outside
        new_c = jnp.where(curvatures == 0, 0.0, 1.0 / (delta_n * new_f))

    return _update_at_stage(
        telescope, stage, lambda g: g.surface.curvatures, new_c
    )


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
    new_groups = [
        g for i, g in enumerate(telescope.obstruction_groups) if i != group_idx
    ]
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

    stages_info = []
    for s in telescope.stage_indices():
        g = telescope.stage(s)
        if isinstance(g.aperture, DiskAperture):
            ap = "disk"
        elif isinstance(g.aperture, PolygonAperture):
            ap = "polygon"
        else:
            ap = "unknown"
        stages_info.append(
            {"stage": s, "kind": g.kind, "n_elements": g.n_elements, "aperture": ap}
        )

    if telescope.optical_groups:
        all_positions = jnp.concatenate(
            [g.positions for g in telescope.optical_groups], axis=0
        )
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