"""Mirror facets and templates."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ...core.apertures import Aperture
from ...core.interactions import ReflectInteraction
from ...core.optics import OpticalElementGroup
from ..schemas import (
    AsphericSurfaceSchema,
    BSDFSchema,
    CircularApertureSchema,
    MirrorSchema,
    MirrorTemplateSchema,
    PolygonApertureSchema,
    TabulatedCurveSchema,
    ZernikeSurfaceSchema,
)
from ._common import (
    _aperture_from_schemas,
    _aperture_to_schema,
    _bucket_by_aperture_signature,
    _pad_aspherics,
    _strip_trailing_zeros,
    _to_float_list,
)
from .bsdf import _bsdf_to_schema, _build_bsdf_for_bucket
from .curves import _build_curve_for_bucket, _curve_schema_to_key, _curve_to_schema
from .surfaces import (
    _asphere_surface_arrays,
    _compose_surface,
    _split_surface,
    _surface_spec_key,
    _zernike_to_schema,
)

if TYPE_CHECKING:
    pass


class _ParsedMirror(NamedTuple):
    """Resolved mirror data with template overrides applied."""

    position: list[float]
    orientation: list[float]
    curvature: float
    conic: float
    aspheric: list[float]
    has_aspheric: bool
    offset: list[float]
    stage: int
    aperture: CircularApertureSchema | PolygonApertureSchema
    bsdf: BSDFSchema | None
    reflectivity: float
    reflectivity_curve: TabulatedCurveSchema | None
    zernike: ZernikeSurfaceSchema | None


class _ResolvedSurface(NamedTuple):
    curvature: float
    conic: float
    aspheric: list[float]
    has_aspheric: bool
    zernike: ZernikeSurfaceSchema | None


def _resolve_surface(
    mirror: MirrorSchema, template: MirrorTemplateSchema | None
) -> _ResolvedSurface:
    """Resolve a mirror's surface: the mirror is the joint of itself and its
    (optional) template, field by field, with the mirror's own value winning
    whenever both define it.

    curvature / conic / aspheric / zernike are each resolved
    independently -- a mirror may override just one (e.g. curvature for a
    segmented primary panel) while inheriting the rest from the template, or
    a template-less mirror may set all of them itself, or a mirror may supply
    its own zernike (e.g. a measured per-panel figure error) while
    sharing the template's aspheric base with every other panel.
    """
    asph = zern = None
    if template is not None and template.surface is not None:
        asph, zern = _split_surface(template.surface)
    override = (
        mirror.curvature is not None or mirror.conic is not None or mirror.aspheric is not None
    )
    base_c = asph.curvature if asph is not None else 0.0
    base_k = asph.conic if asph is not None else 0.0
    base_a = asph.aspheric if asph is not None else []
    return _ResolvedSurface(
        curvature=mirror.curvature if mirror.curvature is not None else base_c,
        conic=mirror.conic if mirror.conic is not None else base_k,
        aspheric=mirror.aspheric if mirror.aspheric is not None else base_a,
        has_aspheric=asph is not None or override,
        zernike=mirror.zernike if mirror.zernike is not None else zern,
    )


def _resolve_bsdf(
    mirror: MirrorSchema,
    template: MirrorTemplateSchema | None,
) -> BSDFSchema | None:
    """Resolve the per-mirror BSDF schema (mirror overrides template)."""
    if mirror.bsdf is not None:
        return mirror.bsdf
    return template.bsdf if template is not None else None


def _resolve_reflectivity(
    mirror: MirrorSchema,
    template: MirrorTemplateSchema | None,
) -> tuple[float, TabulatedCurveSchema | None]:
    """Resolve (reflectivity, reflectivity_curve) from mirror + template.

    Both halves of the pair follow the same mirror-wins-if-defined rule as
    every other joint field.
    """
    template_scalar = (
        template.reflectivity if template is not None and template.reflectivity is not None else 1.0
    )
    scalar = mirror.reflectivity if mirror.reflectivity is not None else float(template_scalar)
    curve = (
        mirror.reflectivity_curve
        if mirror.reflectivity_curve is not None
        else (template.reflectivity_curve if template is not None else None)
    )
    return float(scalar), curve


def mirrors_from_schemas(
    mirrors: list[MirrorSchema],
    templates: dict[str, MirrorTemplateSchema],
    n_samples: int,
    *,
    key: Array,
) -> list[OpticalElementGroup]:
    """Convert validated mirror schemas to OpticalElementGroup domain objects.

    Groups mirrors by (stage, aperture_type, vertex_count), then constructs
    OpticalElementGroup directly with composable modules.
    """
    if not mirrors:
        return []

    # Resolve templates and parse each mirror into typed tuples
    parsed: list[_ParsedMirror] = []
    for mirror in mirrors:
        # A mirror without a template is fully self-contained; TelescopeConfigSchema
        # already validated that a non-None template name exists in templates.
        template = templates[mirror.template] if mirror.template is not None else None
        surface = _resolve_surface(mirror, template)
        bsdf = _resolve_bsdf(mirror, template)
        refl, refl_curve = _resolve_reflectivity(mirror, template)

        parsed.append(
            _ParsedMirror(
                position=mirror.position,
                orientation=mirror.orientation,
                curvature=surface.curvature,
                conic=surface.conic,
                aspheric=surface.aspheric,
                has_aspheric=surface.has_aspheric,
                offset=mirror.offset,
                stage=mirror.stage,
                aperture=mirror.aperture,
                bsdf=bsdf,
                reflectivity=refl,
                reflectivity_curve=refl_curve,
                zernike=surface.zernike,
            )
        )

    groups: list[OpticalElementGroup] = []

    # Group by stage
    by_stage: dict[int, list[_ParsedMirror]] = defaultdict(list)
    for p in parsed:
        by_stage[p.stage].append(p)

    for stage, stage_mirrors in sorted(by_stage.items()):
        for bucket in _bucket_by_aperture_signature(stage_mirrors, lambda m: m.aperture):
            aperture = _aperture_from_schemas([m.aperture for m in bucket])
            key, subkey = jax.random.split(key)
            groups.append(
                _build_mirror_group(bucket, aperture, stage, n_samples, sample_key=subkey)
            )

    return groups


def _build_mirror_group(
    mirrors: list[_ParsedMirror],
    aperture: Aperture,
    stage: int,
    n_samples: int,
    *,
    sample_key: Array,
) -> OpticalElementGroup:
    """Build OpticalElementGroup from parsed mirrors and a pre-built aperture.

    Thin adapter that projects parsed schema data into arrays and delegates
    to iactrace.telescope.mirrors.mirror_group, the canonical
    reflective-group builder for the whole project.
    """
    from ...telescope.mirrors import mirror_group

    n_elements = len(mirrors)

    bsdf = _build_bsdf_for_bucket([m.bsdf for m in mirrors])

    reflectivities = jnp.asarray([m.reflectivity for m in mirrors])
    curve = _build_curve_for_bucket(
        [m.reflectivity_curve for m in mirrors],
        n_elements,
    )
    group = mirror_group(
        positions=jnp.asarray([m.position for m in mirrors]),
        rotations=jnp.asarray([m.orientation for m in mirrors]),
        curvatures=jnp.asarray([m.curvature for m in mirrors]),
        conics=jnp.asarray([m.conic for m in mirrors]),
        aspherics=_pad_aspherics([m.aspheric for m in mirrors]),
        offsets=jnp.asarray([m.offset for m in mirrors]),
        aperture=aperture,
        reflectivity=reflectivities,
        reflectivity_curve=curve,
        bsdf=bsdf,
        sample_key=sample_key,
        optical_stage=stage,
        n_samples=n_samples,
    )
    return _compose_surface(
        group,
        [m.zernike for m in mirrors],
        has_aspheric=any(m.has_aspheric for m in mirrors),
    )


class _MirrorData(NamedTuple):
    """One mirror element's resolved fields, before the template/self-contained
    decision (mirrors_to_schemas).
    """

    group: OpticalElementGroup
    i: int
    asph_schema: AsphericSurfaceSchema | None
    zern_schema: ZernikeSurfaceSchema | None
    curve_schema: TabulatedCurveSchema | None
    offset: Array
    bsdf_schema: BSDFSchema | None
    reflectivity: float


def _mirror_base_key(d: _MirrorData) -> tuple | None:
    """Dedup key for a mirror's templatable fields (aspheric base + curve).

    None when the mirror has no aspheric base at all (a standalone
    Zernike surface); such mirrors never join a template. zernike and
    bsdf are deliberately excluded -- they stay per-mirror even when the
    aspheric base is shared (see mirrors_to_schemas).
    """
    if d.asph_schema is None:
        return None
    return (_surface_spec_key(d.asph_schema), _curve_schema_to_key(d.curve_schema))


def mirrors_to_schemas(
    groups: list[OpticalElementGroup],
) -> tuple[dict[str, MirrorTemplateSchema], list[MirrorSchema]]:
    """Extract mirror schemas from OpticalElementGroup list.

    Each mirror is written as the joint of an optional template and its own
    fields, mirroring how loading resolves them (_resolve_surface):

    - The aspheric base (curvature/conic/aspheric) plus reflectivity_curve
      are deduplicated into a shared template when two or more mirrors have the
      exact same combination. A mirror whose combination is unique to it (or
      has no aspheric base at all -- a standalone Zernike surface) gets no
      template: its curvature/conic/aspheric/reflectivity_curve are written
      directly.
    - zernike is always written directly on the mirror, never folded into
      a template, since it typically represents a per-panel measured figure
      error even when every panel shares the same base prescription.
    - bsdf is always per-mirror, as before.
    """
    data: list[_MirrorData] = []
    for group in groups:
        match group.interaction_module:
            case ReflectInteraction() as interaction:
                pass
            case _:
                continue

        curve_schema = _curve_to_schema(interaction.reflectivity_curve)
        asph, zern, offsets = _asphere_surface_arrays(group.surface, len(group))

        for i in range(len(group)):
            asph_schema = (
                AsphericSurfaceSchema(
                    curvature=float(asph.curvatures[i]),
                    conic=float(asph.conics[i]),
                    aspheric=_strip_trailing_zeros(_to_float_list(asph.aspherics[i])),
                )
                if asph is not None
                else None
            )
            data.append(
                _MirrorData(
                    group=group,
                    i=i,
                    asph_schema=asph_schema,
                    zern_schema=_zernike_to_schema(zern, i),
                    curve_schema=curve_schema,
                    offset=offsets[i],
                    bsdf_schema=_bsdf_to_schema(group.bsdf, i),
                    reflectivity=float(interaction.reflectivity[i]),
                )
            )

    counts: dict[tuple, int] = defaultdict(int)
    for d in data:
        key = _mirror_base_key(d)
        if key is not None:
            counts[key] += 1

    templates: dict[str, MirrorTemplateSchema] = {}
    key_to_template: dict[tuple, str] = {}
    mirrors: list[MirrorSchema] = []

    for d in data:
        key = _mirror_base_key(d)

        if key is not None and counts[key] > 1:
            template_name = key_to_template.get(key)
            if template_name is None:
                template_name = f"template_{len(templates)}"
                key_to_template[key] = template_name
                templates[template_name] = MirrorTemplateSchema(
                    surface=d.asph_schema,
                    reflectivity_curve=d.curve_schema,
                )
            curvature = conic = aspheric = None
            reflectivity_curve = None
        else:
            template_name = None
            reflectivity_curve = d.curve_schema
            if d.asph_schema is None:
                curvature = conic = aspheric = None
            else:
                curvature = d.asph_schema.curvature
                conic = d.asph_schema.conic if d.asph_schema.conic != 0.0 else None
                aspheric = d.asph_schema.aspheric or None

        mirrors.append(
            MirrorSchema(
                position=_to_float_list(d.group.positions[d.i]),
                orientation=_to_float_list(d.group.rotations[d.i]),
                aperture=_aperture_to_schema(d.group.aperture, d.i),
                template=template_name,
                curvature=curvature,
                conic=conic,
                aspheric=aspheric,
                zernike=d.zern_schema,
                stage=d.group.optical_stage,
                offset=_to_float_list(d.offset),
                bsdf=d.bsdf_schema,
                reflectivity=(d.reflectivity if d.reflectivity != 1.0 else None),
                reflectivity_curve=reflectivity_curve,
                id=f"M_{len(mirrors)}",
            )
        )

    return templates, mirrors
