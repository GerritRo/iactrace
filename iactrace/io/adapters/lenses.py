"""Refracting surfaces and plano slabs (windows), both directions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ...core.apertures import Aperture
from ...core.interactions import RefractInteraction, SlabInteraction
from ...core.optics import OpticalElementGroup
from ...core.surfaces import AsphericSurfaceGroup, ZernikeSurfaceGroup
from ..schemas import AsphericDiskLensSchema, PlanoSlabSchema
from ._common import (
    _aperture_from_schemas,
    _aperture_to_schema,
    _bucket_by_aperture_signature,
    _pad_aspherics,
    _to_float_list,
)
from .curves import (
    _build_curve_for_bucket,
    _build_index_for_bucket,
    _curve_to_schema,
    _index_to_schema,
)
from .surfaces import (
    _asphere_surface_arrays,
    _compose_surface,
    _split_surface,
    _surface_components,
    _surface_to_spec,
)

if TYPE_CHECKING:
    pass

LensSchemaType = AsphericDiskLensSchema | PlanoSlabSchema


def lenses_from_schemas(
    lenses: list[LensSchemaType],
    *,
    key: Array,
) -> list[OpticalElementGroup]:
    """Convert validated lens schemas to OpticalElementGroup domain objects.

    Buckets by type via _LENS_SPECS (mirroring how obstructions
    dispatch on _OBSTRUCTION_SPECS), then within each type bucket groups
    by (stage, aperture_signature) via _build_lens_groups_by_stage,
    mirroring how mirrors_from_schemas groups mirrors.
    """
    by_type: dict[str, list] = defaultdict(list)
    for lens in lenses:
        by_type[lens.type].append(lens)

    groups: list[OpticalElementGroup] = []
    for spec in _LENS_SPECS:
        bucket = by_type.get(spec.type_name)
        if bucket:
            key, groups = _build_lens_groups_by_stage(bucket, spec.builder, key, groups)
    return groups


def _build_lens_groups_by_stage[L: AsphericDiskLensSchema | PlanoSlabSchema](
    lenses: list[L],
    builder: Callable[[list[L], Aperture, int, Array], OpticalElementGroup],
    key: Array,
    groups: list[OpticalElementGroup],
) -> tuple[Array, list[OpticalElementGroup]]:
    """Bucket lenses by (stage, aperture signature) and build one group per bucket."""
    by_stage: dict[int, list[L]] = defaultdict(list)
    for lens in lenses:
        by_stage[lens.stage].append(lens)
    for stage, lens_list in by_stage.items():
        for bucket in _bucket_by_aperture_signature(lens_list, lambda lens: lens.aperture):
            aperture = _aperture_from_schemas([lens.aperture for lens in bucket])
            key, subkey = jax.random.split(key)
            groups.append(builder(bucket, aperture, stage, subkey))
    return key, groups


def _build_aspheric_disk_lens_group(
    lenses: list[AsphericDiskLensSchema],
    aperture: Aperture,
    stage: int,
    sample_key: Array,
) -> OpticalElementGroup:
    """Build an aspheric-disk refractive group via the telescope helper.

    Delegates to iactrace.telescope.lenses.refractive_group once
    schema fields have been projected into arrays.
    """
    from ...telescope.lenses import refractive_group

    n = len(lenses)
    curve = _build_curve_for_bucket(
        [lens.transmittance_curve for lens in lenses],
        n,
    )
    split = [_split_surface(lens.surface) for lens in lenses]  # (aspheric, zernike) per lens

    group = refractive_group(
        positions=jnp.asarray([lens.position for lens in lenses]),
        rotations=jnp.asarray([lens.orientation for lens in lenses]),
        curvatures=jnp.asarray([a.curvature if a else 0.0 for a, _ in split]),
        conics=jnp.asarray([a.conic if a else 0.0 for a, _ in split]),
        aspherics=_pad_aspherics([a.aspheric if a else [] for a, _ in split]),
        offsets=jnp.asarray([lens.offset for lens in lenses]),
        aperture=aperture,
        index=_build_index_for_bucket([lens.index for lens in lenses], n),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        transmittance_curve=curve,
        sample_key=sample_key,
        optical_stage=stage,
    )
    return _compose_surface(
        group,
        [z for _, z in split],
        has_aspheric=any(a is not None for a, _ in split),
    )


def _build_plano_slab_group(
    lenses: list[PlanoSlabSchema],
    aperture: Aperture,
    stage: int,
    sample_key: Array,
) -> OpticalElementGroup:
    """Build a plano-slab group via the telescope helper.

    Delegates to iactrace.telescope.lenses.slab_group once schema
    fields have been projected into arrays.
    """
    from ...telescope.lenses import slab_group

    n = len(lenses)
    curve = _build_curve_for_bucket(
        [lens.transmittance_curve for lens in lenses],
        n,
    )

    return slab_group(
        positions=jnp.asarray([lens.position for lens in lenses]),
        rotations=jnp.asarray([lens.orientation for lens in lenses]),
        aperture=aperture,
        index=_build_index_for_bucket([lens.index for lens in lenses], n),
        thickness=jnp.asarray([lens.thickness for lens in lenses]),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        transmittance_curve=curve,
        sample_key=sample_key,
        optical_stage=stage,
    )


def lenses_to_schemas(
    groups: list[OpticalElementGroup] | None,
) -> list[LensSchemaType]:
    """Extract lens schemas from OpticalElementGroup list; see _LENS_SPECS."""
    if not groups:
        return []

    lenses: list[LensSchemaType] = []
    for group in groups:
        spec = _LENS_SPEC_BY_INTERACTION.get(type(group.interaction_module))
        if spec is None:
            continue
        lenses.extend(spec.extract_group(group, len(lenses)))
    return lenses


def _extract_aspheric_disk_lens(
    group: OpticalElementGroup,
    interaction: RefractInteraction,
    i: int,
    counter: int,
    asph: AsphericSurfaceGroup | None,
    zern: ZernikeSurfaceGroup | None,
    offsets: Array,
) -> AsphericDiskLensSchema:
    """Extract an AsphericDiskLensSchema from element i of a group."""
    curve_schema = _curve_to_schema(interaction.transmittance_curve)
    return AsphericDiskLensSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        surface=_surface_to_spec(asph, zern, i),
        index=_index_to_schema(interaction.index, i),
        offset=_to_float_list(offsets[i]),
        transmittance=float(interaction.transmittance[i]),
        transmittance_curve=curve_schema,
        stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def _extract_plano_slab_lens(
    group: OpticalElementGroup,
    interaction: SlabInteraction,
    i: int,
    counter: int,
) -> PlanoSlabSchema:
    """Extract a PlanoSlabSchema from element i of a group."""
    curve_schema = _curve_to_schema(interaction.transmittance_curve)
    return PlanoSlabSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        thickness=float(interaction.thickness[i]),
        index=_index_to_schema(interaction.index, i),
        transmittance=float(interaction.transmittance[i]),
        transmittance_curve=curve_schema,
        stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def _extract_aspheric_disk_lenses(
    group: OpticalElementGroup, start: int
) -> list[AsphericDiskLensSchema]:
    """Extract every element of an aspheric-disk lens group, starting at index start."""
    interaction = group.interaction_module
    # _LENS_SPEC_BY_INTERACTION only ever routes here for a RefractInteraction
    # group; the assert both documents and narrows that for the type checker.
    assert isinstance(interaction, RefractInteraction)
    asph, zern, offsets = _asphere_surface_arrays(group.surface, len(group))
    return [
        _extract_aspheric_disk_lens(group, interaction, i, start + i, asph, zern, offsets)
        for i in range(len(group))
    ]


def _extract_plano_slab_lenses(group: OpticalElementGroup, start: int) -> list[PlanoSlabSchema]:
    """Extract every element of a plano-slab group, starting at index start."""
    interaction = group.interaction_module
    # _LENS_SPEC_BY_INTERACTION only ever routes here for a SlabInteraction
    # group; the assert both documents and narrows that for the type checker.
    assert isinstance(interaction, SlabInteraction)
    _, slab_zern = _surface_components(group.surface)
    if slab_zern is not None:
        raise ValueError("cannot serialise a Zernike figure error on a plano slab")
    return [_extract_plano_slab_lens(group, interaction, i, start + i) for i in range(len(group))]


class _LensSpec(NamedTuple):
    """Bidirectional spec for one lens type, mirroring _ObsSpec.

    builder constructs one bucket's OpticalElementGroup (see
    _build_lens_groups_by_stage); extract_group is its inverse,
    projecting a whole group back to its per-element schemas. Each lens kind
    needs different per-group setup before its per-element loop (an aspheric
    disk resolves its surface decomposition once per group; a slab is always
    flat and only checks for a stray Zernike term), so unlike an obstruction's
    flat field table, that setup lives inside each kind's own function.
    """

    type_name: str
    schema: type
    interaction: type
    builder: Callable[[list, Aperture, int, Array], OpticalElementGroup]
    extract_group: Callable[[OpticalElementGroup, int], list]


# The single source of truth for lens round-tripping; see _OBSTRUCTION_SPECS /
# _CONCENTRATOR_SPECS. Adding a lens type is one entry here plus its schema
# (io.schemas) and builder/extractor functions.
_LENS_SPECS: tuple[_LensSpec, ...] = (
    _LensSpec(
        "aspheric_disk",
        AsphericDiskLensSchema,
        RefractInteraction,
        _build_aspheric_disk_lens_group,
        _extract_aspheric_disk_lenses,
    ),
    _LensSpec(
        "plano_slab",
        PlanoSlabSchema,
        SlabInteraction,
        _build_plano_slab_group,
        _extract_plano_slab_lenses,
    ),
)
_LENS_SPEC_BY_INTERACTION: dict[type, _LensSpec] = {s.interaction: s for s in _LENS_SPECS}
