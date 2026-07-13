from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..camera.detector import PMT, ConstantQE
from ..camera.optics import OkumuraCone, WinstonCone
from ..camera.optics.winston import cpc_full_length
from ..camera.sensor_group import HexagonalSensorGroup, SquareSensorGroup
from ..core.apertures import Aperture, DiskAperture, PolygonAperture
from ..core.bsdf import BSDF, DoubleGaussianBSDF, GaussianBSDF
from ..core.coatings import Coating, TabulatedCoating
from ..core.interactions import (
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
)
from ..core.obstructions import (
    BoxGroup,
    CylinderGroup,
    ObstructionGroup,
    OpenCylinderGroup,
    OrientedBoxGroup,
    SphereGroup,
    TriangleGroup,
)
from ..core.optics import OpticalElementGroup
from ..core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from ..core.transforms import euler_to_matrix
from .schemas import (
    AsphericDiskLensSchema,
    AsphericSurfaceSchema,
    BoxObstructionSchema,
    BSDFSchema,
    CameraFileSchema,
    CircularApertureSchema,
    ConcentratorSchema,
    ConstantQESchema,
    CylinderObstructionSchema,
    DoubleGaussianBSDFSchema,
    GaussianBSDFSchema,
    HexagonalSensorSchema,
    MirrorSchema,
    MirrorTemplateSchema,
    OkumuraConeSchema,
    OpenCylinderObstructionSchema,
    OrientedBoxObstructionSchema,
    PhotoDetectorSchema,
    PlanoSlabSchema,
    PMTSchema,
    PolygonApertureSchema,
    SphereObstructionSchema,
    SquareSensorSchema,
    TabulatedCurveSchema,
    TelescopeConfigSchema,
    TelescopeMetadataSchema,
    TriangleObstructionSchema,
    WinstonConeSchema,
    ZernikeSurfaceSchema,
)

if TYPE_CHECKING:
    from ..camera import Camera
    from ..camera.detection_chain import DetectionChain
    from ..camera.detector import PhotoDetector
    from ..camera.optics import Concentrator
    from ..camera.sensor_group import SensorGroup
    from ..telescope import Telescope

# Type aliases
LensSchemaType = AsphericDiskLensSchema | PlanoSlabSchema
ObstructionSchemaType = (
    CylinderObstructionSchema
    | OpenCylinderObstructionSchema
    | BoxObstructionSchema
    | SphereObstructionSchema
    | OrientedBoxObstructionSchema
    | TriangleObstructionSchema
)
SensorSchemaType = SquareSensorSchema | HexagonalSensorSchema


# Helpers


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
    reflectivity_scalar: float
    coating_curve: TabulatedCurveSchema | None
    zernike: ZernikeSurfaceSchema | None


def _to_float_list(arr: np.ndarray | jnp.ndarray) -> list[float]:
    """Convert a JAX/NumPy array to a plain list of floats."""
    return [float(x) for x in np.asarray(arr)]


def _strip_trailing_zeros(values: list[float]) -> list[float]:
    """Strip trailing zero coefficients from an aspheric list."""
    while values and values[-1] == 0.0:
        values.pop()
    return values


def _pad_aspherics(aspheric_list: list[list[float]]) -> jnp.ndarray:
    """Pad aspheric coefficient arrays to uniform length.

    When all elements have empty coefficient lists, returns shape (N, 0)
    so that sag_raw skips the aspheric computation entirely.
    """
    if not aspheric_list:
        return jnp.zeros((0, 0))

    max_len = max(len(a) for a in aspheric_list)

    if max_len == 0:
        return jnp.zeros((len(aspheric_list), 0))

    padded = []
    for a in aspheric_list:
        arr = jnp.asarray(a)
        if len(arr) < max_len:
            arr = jnp.concatenate([arr, jnp.zeros(max_len - len(arr))])
        padded.append(arr)

    return jnp.stack(padded)


def _ensure_ccw(vertices: jnp.ndarray) -> jnp.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    vx, vy = vertices[:, 0], vertices[:, 1]
    signed_area = 0.5 * jnp.sum(vx * jnp.roll(vy, -1) - jnp.roll(vx, -1) * vy)
    return jnp.where(signed_area < 0, vertices[::-1], vertices)


def _disk_aperture_from_schemas(schemas: list[CircularApertureSchema]) -> DiskAperture:
    return DiskAperture(
        radii=jnp.asarray([s.radius for s in schemas]),
        inner_radii=jnp.asarray([s.inner_radius for s in schemas]),
    )


def _polygon_aperture_from_schemas(schemas: list[PolygonApertureSchema]) -> PolygonAperture:
    vertices = jnp.stack([_ensure_ccw(jnp.asarray(s.vertices)) for s in schemas])
    return PolygonAperture(vertices=vertices, n_vertices=int(vertices.shape[1]))


def _aperture_from_schemas(
    schemas: list[CircularApertureSchema | PolygonApertureSchema],
) -> Aperture:
    """Build a single ``Aperture`` from a list of homogeneous aperture schemas.

    All schemas must share the same kind (and, for polygons, the same
    vertex count); callers are expected to bucket via
    :func:`_bucket_by_aperture_signature` first.
    """
    disks: list[CircularApertureSchema] = []
    polys: list[PolygonApertureSchema] = []
    for s in schemas:
        match s.type:
            case "circular":
                disks.append(s)
            case "polygon":
                polys.append(s)
    if disks and not polys:
        return _disk_aperture_from_schemas(disks)
    if polys and not disks:
        return _polygon_aperture_from_schemas(polys)
    raise ValueError("aperture schemas must be homogeneous (all disk or all polygon)")


def _bucket_by_aperture_signature[T](
    items: list[T],
    aperture_of: Callable[[T], CircularApertureSchema | PolygonApertureSchema],
) -> list[list[T]]:
    """Group items so each bucket has a single aperture signature.

    Disk apertures form one bucket; each distinct polygon vertex count
    forms its own bucket. Order within a bucket follows input order.
    """
    disk_bucket: list[T] = []
    poly_buckets: dict[int, list[T]] = defaultdict(list)
    for item in items:
        ap = aperture_of(item)
        match ap.type:
            case "polygon":
                poly_buckets[len(ap.vertices)].append(item)
            case "circular":
                disk_bucket.append(item)

    buckets: list[list[T]] = []
    if disk_bucket:
        buckets.append(disk_bucket)
    buckets.extend(poly_buckets.values())
    return buckets


def _surface_list(spec) -> list:
    """Normalise a surface spec (a single shape or a list) to a list of shapes."""
    return list(spec) if isinstance(spec, list) else [spec]


def _split_surface(
    spec,
) -> tuple[AsphericSurfaceSchema | None, ZernikeSurfaceSchema | None]:
    """Split a surface spec into its ``(aspheric, zernike)`` shapes.

    At most one of each is allowed today; the surface's sag is their sum.
    """
    asph: AsphericSurfaceSchema | None = None
    zern: ZernikeSurfaceSchema | None = None
    for s in _surface_list(spec):
        match s.type:
            case "aspheric":
                if asph is not None:
                    raise ValueError("a surface may list at most one aspheric shape")
                asph = s
            case "zernike":
                if zern is not None:
                    raise ValueError("a surface may list at most one zernike shape")
                zern = s
    return asph, zern


def _single_element_surface(spec) -> AsphericSurfaceGroup | ZernikeSurfaceGroup | SumSurfaceGroup:
    """Build a one-element (N=1) core surface from a schema spec.

    Reuses :func:`_split_surface` -- the same aspheric/zernike decomposition
    every mirror and lens surface goes through -- so any surface an optical
    element can describe (a bare aspheric shape, a bare zernike shape, or
    their sum) is buildable from a single spec. The N-wide counterpart lives
    in :func:`_build_mirror_group` / :func:`_build_aspheric_disk_lens_group`
    (batched over a bucket); this is the N=1 case, used by
    :func:`_pmt_from_schema`.
    """
    asph, zern = _split_surface(spec)
    aspheric = AsphericSurfaceGroup(
        offsets=jnp.zeros((1, 2)),
        curvatures=jnp.asarray([asph.curvature if asph is not None else 0.0]),
        conics=jnp.asarray([asph.conic if asph is not None else 0.0]),
        aspherics=_pad_aspherics([asph.aspheric if asph is not None else []]),
    )
    if zern is None:
        return aspheric
    zernike = ZernikeSurfaceGroup(coeffs=jnp.asarray([zern.coeffs]), r_norm=jnp.asarray([zern.r_norm]))
    if asph is None:
        return zernike
    return SumSurfaceGroup([aspheric, zernike])


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

    ``curvature`` / ``conic`` / ``aspheric`` / ``zernike`` are each resolved
    independently -- a mirror may override just one (e.g. ``curvature`` for a
    segmented primary panel) while inheriting the rest from the template, or
    a template-less mirror may set all of them itself, or a mirror may supply
    its own ``zernike`` (e.g. a measured per-panel figure error) while
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
    """Resolve (bulk_scalar, coating_curve) from mirror + template.

    Both the scalar and the coating follow the same mirror-wins-if-defined
    rule as every other joint field.
    """
    template_scalar = (
        template.reflectivity if template is not None and template.reflectivity is not None else 1.0
    )
    scalar = mirror.reflectivity if mirror.reflectivity is not None else float(template_scalar)
    coating = mirror.coating if mirror.coating is not None else (
        template.coating if template is not None else None
    )
    return float(scalar), coating


def _curves_equal(
    a: TabulatedCurveSchema,
    b: TabulatedCurveSchema,
) -> bool:
    """Structural equality of two tabulated curve schemas."""
    return a.angles_deg == b.angles_deg and a.values == b.values


def _build_coating_for_bucket(
    curves: list[TabulatedCurveSchema | None],
    n_elements: int,
) -> Coating | None:
    """Resolve a list of per-element curve schemas into a single coating.

    All ``None`` -> ``None`` (caller's default physics applies).
    One distinct curve -> broadcast across all elements.
    Coated mixed with uncoated, or several distinct curves -> ``ValueError``.
    """
    distinct: list[TabulatedCurveSchema] = []
    for c in curves:
        if c is None:
            continue
        if not any(_curves_equal(c, d) for d in distinct):
            distinct.append(c)

    if not distinct:
        return None
    if any(c is None for c in curves):
        raise ValueError(
            "Elements grouped at the same stage with the same aperture "
            "must either all define a `coating` or all omit it; mixing "
            "coated and uncoated elements would silently apply one "
            "element's coating to the rest. Split them across stages or "
            "harmonize their `coating` fields."
        )
    if len(distinct) > 1:
        raise ValueError(
            "Elements grouped at the same stage with the same aperture "
            "must resolve to a single coating, but multiple distinct "
            "coating curves were found; broadcasting one would silently "
            "apply it to the rest. Split them across stages or harmonize "
            "their `coating` fields."
        )

    curve = distinct[0]
    return TabulatedCoating.from_degrees(
        angles_deg=curve.angles_deg,
        values=curve.values,
        n_elements=n_elements,
    )


def _coating_to_curve_schema(
    coating: Coating | None,
) -> TabulatedCurveSchema | None:
    """Project a Coating to a serialisable curve, or ``None`` if trivial.

    ``None`` and :class:`ConstantCoating` round-trip as ``None`` so
    existing YAMLs stay byte-identical. A :class:`TabulatedCoating`
    emits the inline ``{type: table, ...}`` form. The YAML schema holds
    one shared curve per template, so a per-element coating (rows that
    differ across elements) raises :class:`ValueError` rather than
    silently serialising only the first element's row.

    Raises:
        ValueError: If ``coating`` is a per-element
            :class:`TabulatedCoating` whose rows are not all equal.
    """
    if isinstance(coating, TabulatedCoating):
        value_rows = np.asarray(coating.values)
        # YAML expresses one shared curve per template. A per-element
        # coating (rows differ) cannot be represented; fail loudly rather
        # than silently serialising only element 0's row. Mirrors the
        # loader guard in _build_coating_for_bucket.
        if value_rows.shape[0] > 1 and not np.allclose(value_rows, value_rows[0]):
            raise ValueError(
                "Cannot serialise a per-element TabulatedCoating to YAML: "
                "all elements in a group must share one curve. Split them "
                "across groups, or harmonise their rows before saving."
            )
        cos_table = np.asarray(coating.cos_table)
        values_row = value_rows[0]
        order = np.argsort(-cos_table)  # cos descending -> angles ascending
        angles_deg = [float(x) for x in np.degrees(np.arccos(cos_table[order]))]
        values = [float(x) for x in values_row[order]]
        return TabulatedCurveSchema(angles_deg=angles_deg, values=values)
    return None


def _curve_schema_to_key(
    schema: TabulatedCurveSchema | None,
) -> tuple | None:
    """Hashable key used by ``mirrors_to_schemas`` to dedup templates."""
    if schema is None:
        return None
    return ("table", tuple(schema.angles_deg), tuple(schema.values))


def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to Euler angles (degrees)."""
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        rx = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = 0.0
    return [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]


# Schema -> Domain (loading)


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
        # already validated that a non-None template name exists in `templates`.
        template = templates[mirror.template] if mirror.template is not None else None
        surface = _resolve_surface(mirror, template)
        bsdf = _resolve_bsdf(mirror, template)
        refl_scalar, coating_curve = _resolve_reflectivity(mirror, template)

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
                reflectivity_scalar=refl_scalar,
                coating_curve=coating_curve,
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


class _BsdfSpec(NamedTuple):
    """Bidirectional spec for one BSDF type; see ``_ConcentratorSpec``.

    ``build`` assembles a whole-bucket domain BSDF from a list of per-element
    schemas (already known to be homogeneous and non-empty); ``to_schema``
    projects element ``i`` of a domain BSDF back to a schema, or ``None`` for
    a trivially-zero element.
    """

    type_name: str
    schema: type
    group: type
    build: Callable[[list], BSDF | None]
    # Each entry's to_schema only ever accepts that entry's own BSDF subclass
    # (the driver looks it up by type(bsdf) first), narrower than a plain
    # Callable[[BSDF, int], ...] would allow; typed loosely here for that reason.
    to_schema: Callable[..., BSDFSchema | None]


def _build_gaussian_bsdf(schemas: list[GaussianBSDFSchema | None]) -> GaussianBSDF | None:
    scale = jnp.asarray([s.scale if s is not None else 0.0 for s in schemas])
    if bool(jnp.all(scale == 0)):
        return None
    return GaussianBSDF(scale=scale)


def _gaussian_bsdf_to_schema(bsdf: GaussianBSDF, i: int) -> GaussianBSDFSchema | None:
    scale = float(bsdf.scale[i])
    return None if scale == 0.0 else GaussianBSDFSchema(scale=scale)


def _build_double_gaussian_bsdf(
    schemas: list[DoubleGaussianBSDFSchema | None],
) -> DoubleGaussianBSDF:
    def _col(attr: str) -> Array:
        return jnp.asarray([getattr(s, attr) if s is not None else 0.0 for s in schemas])

    return DoubleGaussianBSDF(
        scale_narrow=_col("scale_narrow"),
        scale_wide=_col("scale_wide"),
        mix_weight=_col("mix_weight"),
    )


def _double_gaussian_bsdf_to_schema(bsdf: DoubleGaussianBSDF, i: int) -> DoubleGaussianBSDFSchema:
    return DoubleGaussianBSDFSchema(
        scale_narrow=float(bsdf.scale_narrow[i]),
        scale_wide=float(bsdf.scale_wide[i]),
        mix_weight=float(bsdf.mix_weight[i]),
    )


# The single source of truth for BSDF round-tripping; see _CONCENTRATOR_SPECS.
# Adding a BSDF model is one entry here plus a schema variant in
# iactrace.io.schemas.
_BSDF_SPECS: tuple[_BsdfSpec, ...] = (
    _BsdfSpec(
        "gaussian", GaussianBSDFSchema, GaussianBSDF, _build_gaussian_bsdf, _gaussian_bsdf_to_schema
    ),
    _BsdfSpec(
        "double_gaussian",
        DoubleGaussianBSDFSchema,
        DoubleGaussianBSDF,
        _build_double_gaussian_bsdf,
        _double_gaussian_bsdf_to_schema,
    ),
)
_BSDF_SPEC_BY_GROUP: dict[type, _BsdfSpec] = {s.group: s for s in _BSDF_SPECS}
_BSDF_SPEC_BY_TYPE: dict[str, _BsdfSpec] = {s.type_name: s for s in _BSDF_SPECS}


def _build_bsdf_for_bucket(
    schemas: list[BSDFSchema | None],
) -> BSDF | None:
    """Reassemble one group's BSDF from per-element schemas; see ``_BSDF_SPECS``.

    All ``None`` -> ``None`` (perfect specular). Otherwise every element
    that declares a BSDF must share the same ``type``; per-element
    parameters are stacked into the model's arrays, and elements without
    a BSDF default to zero (specular for that element). Mixed types
    raise ``ValueError``, mirroring the per-bucket coating guard in
    :func:`_build_coating_for_bucket`.
    """
    present = [s for s in schemas if s is not None]
    if not present:
        return None

    types = {s.type for s in present}
    if len(types) > 1:
        raise ValueError(
            "Mirrors grouped at the same stage with the same aperture must "
            f"share a single BSDF type; got {sorted(types)}. Split them "
            "across stages, or harmonise their `bsdf.type`."
        )

    spec = _BSDF_SPEC_BY_TYPE.get(present[0].type)
    if spec is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise ValueError(
            f"Unhandled BSDF schema type {present[0].type!r}; add a "
            "_BsdfSpec entry in iactrace.io.adapters."
        )
    return spec.build(schemas)


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
    to :func:`iactrace.telescope.mirrors.mirror_group`, the canonical
    reflective-group builder for the whole project.
    """
    from ..telescope.mirrors import mirror_group

    n_elements = len(mirrors)

    bsdf = _build_bsdf_for_bucket([m.bsdf for m in mirrors])

    reflectivity_scalars = jnp.asarray([m.reflectivity_scalar for m in mirrors])
    coating = _build_coating_for_bucket(
        [m.coating_curve for m in mirrors],
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
        reflectivity=reflectivity_scalars,
        coating=coating,
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


def _build_zernike_for_bucket(
    schemas: list[ZernikeSurfaceSchema | None],
) -> ZernikeSurfaceGroup | None:
    """Reassemble one group's Zernike term from per-element schemas, or ``None``.
    All ``None`` -> ``None`` (no figure error). Otherwise every element's
    coefficients are padded to a common width and stacked; elements without a
    ``zernike`` block contribute zero coefficients (and a placeholder ``r_norm``
    of 1.0, which is irrelevant since their contribution is zero).
    """
    present = [z for z in schemas if z is not None]
    if not present:
        return None
    width = max(len(z.coeffs) for z in present)
    coeffs: list[list[float]] = []
    r_norms: list[float] = []
    for z in schemas:
        if z is None:
            coeffs.append([0.0] * width)
            r_norms.append(1.0)
        else:
            coeffs.append(list(z.coeffs) + [0.0] * (width - len(z.coeffs)))
            r_norms.append(z.r_norm)
    return ZernikeSurfaceGroup(
        coeffs=jnp.asarray(coeffs), r_norm=jnp.asarray(r_norms),
    )


def _compose_surface(
    group: OpticalElementGroup,
    zernike_schemas: list[ZernikeSurfaceSchema | None],
    *,
    has_aspheric: bool,
) -> OpticalElementGroup:
    """Replace the group's built aspheric surface with the composed surface.

    The group is always built with an :class:`AsphericSurfaceGroup` (flat when
    the spec has no aspheric shape). Given the per-element Zernike shapes and
    whether the bucket has an aspheric shape at all:

    - no Zernike -> keep the aspheric surface (bare asphere);
    - Zernike + aspheric -> ``SumSurfaceGroup([asphere, zernike])``;
    - Zernike only -> a standalone :class:`ZernikeSurfaceGroup` (the flat
      placeholder asphere is dropped; its decenter carries over).
    """
    zernike = _build_zernike_for_bucket(zernike_schemas)
    if zernike is None:
        return group
    if not has_aspheric:
        standalone = ZernikeSurfaceGroup(
            coeffs=zernike.coeffs, r_norm=zernike.r_norm, offsets=group.surface.offsets
        )
        return eqx.tree_at(lambda g: g.surface, group, standalone)
    return eqx.tree_at(lambda g: g.surface, group, SumSurfaceGroup([group.surface, zernike]))


def lenses_from_schemas(
    lenses: list[LensSchemaType],
    *,
    key: Array,
) -> list[OpticalElementGroup]:
    """Convert validated lens schemas to OpticalElementGroup domain objects.

    Buckets by ``type`` via ``_LENS_SPECS`` (mirroring how obstructions
    dispatch on ``_OBSTRUCTION_SPECS``), then within each type bucket groups
    by ``(stage, aperture_signature)`` via :func:`_build_lens_groups_by_stage`,
    mirroring how :func:`mirrors_from_schemas` groups mirrors.
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
    """Bucket ``lenses`` by (stage, aperture signature) and build one group per bucket."""
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

    Delegates to :func:`iactrace.telescope.lenses.refractive_group` once
    schema fields have been projected into arrays.
    """
    from ..telescope.lenses import refractive_group

    n = len(lenses)
    coating = _build_coating_for_bucket(
        [lens.coating for lens in lenses],
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
        n_inside=jnp.asarray([lens.n_inside for lens in lenses]),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        coating=coating,
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

    Delegates to :func:`iactrace.telescope.lenses.slab_group` once schema
    fields have been projected into arrays.
    """
    from ..telescope.lenses import slab_group

    n = len(lenses)
    coating = _build_coating_for_bucket(
        [lens.coating for lens in lenses],
        n,
    )

    return slab_group(
        positions=jnp.asarray([lens.position for lens in lenses]),
        rotations=jnp.asarray([lens.orientation for lens in lenses]),
        aperture=aperture,
        n_inside=jnp.asarray([lens.n_inside for lens in lenses]),
        thickness=jnp.asarray([lens.thickness for lens in lenses]),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        coating=coating,
        sample_key=sample_key,
        optical_stage=stage,
    )


class _ObsField(NamedTuple):
    """One field of an obstruction, mapping a schema attr to a group attr.

    ``kind`` selects how the value is projected in each direction:
    ``vec3``/``scalar`` copy through (arrays batch on load, elements read
    back on save); ``euler_matrix`` converts Euler degrees <-> a 3x3 matrix.
    """

    schema_attr: str
    group_attr: str
    kind: str  # 'vec3' | 'scalar' | 'euler_matrix'


class _ObsSpec(NamedTuple):
    """Bidirectional spec for one obstruction primitive type."""

    type_name: str
    schema: type
    group: type
    fields: tuple[_ObsField, ...]


# The single source of truth for obstruction round-tripping. Adding a new
# primitive is one entry here plus its schema (io.schemas) and group
# (core.obstructions) classes; the load/save drivers below are type-agnostic.
_OBSTRUCTION_SPECS: tuple[_ObsSpec, ...] = (
    _ObsSpec(
        "cylinder", CylinderObstructionSchema, CylinderGroup,
        (_ObsField("p1", "p1", "vec3"), _ObsField("p2", "p2", "vec3"), _ObsField("r", "r", "scalar")),
    ),
    _ObsSpec(
        "open_cylinder", OpenCylinderObstructionSchema, OpenCylinderGroup,
        (_ObsField("p1", "p1", "vec3"), _ObsField("p2", "p2", "vec3"), _ObsField("r", "r", "scalar")),
    ),
    _ObsSpec(
        "box", BoxObstructionSchema, BoxGroup,
        (_ObsField("p1", "p1", "vec3"), _ObsField("p2", "p2", "vec3")),
    ),
    _ObsSpec(
        "sphere", SphereObstructionSchema, SphereGroup,
        (_ObsField("center", "centers", "vec3"), _ObsField("r", "radii", "scalar")),
    ),
    _ObsSpec(
        "oriented_box", OrientedBoxObstructionSchema, OrientedBoxGroup,
        (
            _ObsField("center", "centers", "vec3"),
            _ObsField("half_extents", "half_extents", "vec3"),
            _ObsField("rotation", "rotations", "euler_matrix"),
        ),
    ),
    _ObsSpec(
        "triangle", TriangleObstructionSchema, TriangleGroup,
        (_ObsField("v0", "v0", "vec3"), _ObsField("v1", "v1", "vec3"), _ObsField("v2", "v2", "vec3")),
    ),
)


def _build_obstruction_group(spec: _ObsSpec, schemas: list) -> ObstructionGroup:
    """Batch a homogeneous list of obstruction schemas into one group."""
    kwargs: dict[str, object] = {}
    for f in spec.fields:
        values = [getattr(s, f.schema_attr) for s in schemas]
        if f.kind == "euler_matrix":
            kwargs[f.group_attr] = jnp.stack([euler_to_matrix(jnp.asarray(v)) for v in values])
        else:
            kwargs[f.group_attr] = values  # group __init__ applies jnp.asarray
    return spec.group(**kwargs)


def obstructions_from_schemas(
    obstructions: list[ObstructionSchemaType],
) -> list[ObstructionGroup]:
    """Convert validated obstruction schemas to ObstructionGroup domain objects.

    Same-typed schemas are batched into one group. Groups are emitted in
    ``_OBSTRUCTION_SPECS`` declaration order, independent of input order.
    """
    by_type: dict[str, list] = defaultdict(list)
    for obs in obstructions:
        by_type[obs.type].append(obs)
    return [
        _build_obstruction_group(spec, by_type[spec.type_name])
        for spec in _OBSTRUCTION_SPECS
        if by_type.get(spec.type_name)
    ]


def _build_square_group(
    schema: SquareSensorSchema, positions, rotations, concentrator, photodetector, gap
) -> SquareSensorGroup:
    b = schema.bounds
    return SquareSensorGroup(
        positions=positions,
        rotations=rotations,
        width=schema.width,
        height=schema.height,
        bounds=(b[0], b[1], b[2], b[3]),
        edge_width=schema.edge_width,
        concentrator=concentrator,
        photodetector=photodetector,
        gap=gap,
    )


def _build_hex_group(
    schema: HexagonalSensorSchema, positions, rotations, concentrator, photodetector, gap
) -> HexagonalSensorGroup:
    return HexagonalSensorGroup(
        positions=positions,
        rotations=rotations,
        hex_centers=[[x, y] for x, y in zip(schema.centers_x, schema.centers_y, strict=False)],
        edge_width=schema.edge_width,
        concentrator=concentrator,
        photodetector=photodetector,
        gap=gap,
    )


def sensor_from_schema(
    schema: SquareSensorSchema | HexagonalSensorSchema,
) -> SensorGroup:
    """Convert a validated sensor schema to a SensorGroup domain object; see ``_SENSOR_SPECS``.

    The schema ``position`` / ``orientation`` are interpreted as
    **camera-local** coordinates.
    """
    positions = [list(p) for p in schema.positions]
    rotations = [list(r) for r in schema.orientations]
    concentrator = _concentrator_from_schema(schema.concentrator)
    photodetector = _photodetector_from_schema(schema.photodetector)
    spec = _SENSOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise ValueError(f"unknown sensor schema type: {schema.type!r}")
    return spec.build(schema, positions, rotations, concentrator, photodetector, schema.gap)


# Domain -> Schema (saving)


def _bsdf_to_schema(bsdf: BSDF | None, i: int) -> BSDFSchema | None:
    """Project element ``i`` of a group BSDF to a serialisable schema; see ``_BSDF_SPECS``.

    ``None`` and an all-zero :class:`~iactrace.core.bsdf.GaussianBSDF`
    element round-trip as ``None`` so default (specular) mirrors stay
    clean in the YAML. Unhandled BSDF subclasses raise rather than being
    silently dropped to a partial form.
    """
    if bsdf is None:
        return None
    spec = _BSDF_SPEC_BY_GROUP.get(type(bsdf))
    if spec is None:
        raise ValueError(
            f"BSDF type {type(bsdf).__name__} cannot be serialised to "
            "YAML; add a schema variant in iactrace.io.schemas and a "
            "_BsdfSpec entry in iactrace.io.adapters."
        )
    return spec.to_schema(bsdf, i)


def _surface_components(
    surface,
) -> tuple[AsphericSurfaceGroup | None, ZernikeSurfaceGroup | None]:
    """Split a surface into its aspheric and Zernike parts for serialization.
    Accepts a bare :class:`AsphericSurfaceGroup`, a standalone
    :class:`ZernikeSurfaceGroup`, or a :class:`SumSurfaceGroup` composing one of
    each. Either part may be ``None``. Raises if the surface contains anything
    else, more than one of either type, or a non-zero decenter on the composite
    or the Zernike term (the flat per-element schema keeps the decenter on the
    asphere only).
    """
    if isinstance(surface, AsphericSurfaceGroup):
        return surface, None
    if isinstance(surface, ZernikeSurfaceGroup):
        if not np.allclose(np.asarray(surface.offsets), 0.0):
            raise ValueError(
                "cannot serialise a Zernike surface with a non-zero decenter"
            )
        return None, surface
    if isinstance(surface, SumSurfaceGroup):
        if not np.allclose(np.asarray(surface.offsets), 0.0):
            raise ValueError(
                "cannot serialise a SumSurfaceGroup with a non-zero composite "
                "decenter; keep the decenter on the aspheric component"
            )
        asph: AsphericSurfaceGroup | None = None
        zern: ZernikeSurfaceGroup | None = None
        for c in surface.components:
            if isinstance(c, AsphericSurfaceGroup) and asph is None:
                asph = c
            elif isinstance(c, ZernikeSurfaceGroup) and zern is None:
                zern = c
            else:
                raise ValueError(
                    f"cannot serialise a SumSurfaceGroup containing "
                    f"{type(c).__name__}; only one AsphericSurfaceGroup and one "
                    "ZernikeSurfaceGroup are supported"
                )
        if zern is not None and not np.allclose(np.asarray(zern.offsets), 0.0):
            raise ValueError(
                "cannot serialise a Zernike term with a non-zero decenter"
            )
        return asph, zern
    raise ValueError(
        f"cannot serialise surface type {type(surface).__name__}"
    )


def _zernike_to_schema(
    zernike: ZernikeSurfaceGroup | None, i: int
) -> ZernikeSurfaceSchema | None:
    """Project element ``i`` of a Zernike term to a schema, or ``None``.
    Elements whose coefficients are all zero round-trip as ``None`` so default
    (figure-error-free) elements stay clean in the YAML.
    """
    if zernike is None:
        return None
    coeffs = _strip_trailing_zeros(_to_float_list(zernike.coeffs[i]))
    if not coeffs:
        return None
    return ZernikeSurfaceSchema(coeffs=coeffs, r_norm=float(zernike.r_norm[i]))


def _surface_to_spec(asph, zern, i: int):
    """Serialise element ``i``'s surface into a spec: one shape, or a summed list.

    An aspheric shape comes first (it supplies the intersection guess); a
    non-trivial Zernike term follows. A standalone Zernike surface serialises as
    a single ``zernike`` shape.
    """
    shapes: list = []
    if asph is not None:
        shapes.append(
            AsphericSurfaceSchema(
                curvature=float(asph.curvatures[i]),
                conic=float(asph.conics[i]),
                aspheric=_strip_trailing_zeros(_to_float_list(asph.aspherics[i])),
            )
        )
    z = _zernike_to_schema(zern, i)
    if z is not None:
        shapes.append(z)
    if not shapes:
        shapes.append(AsphericSurfaceSchema(curvature=0.0, conic=0.0, aspheric=[]))
    return shapes[0] if len(shapes) == 1 else shapes


def _surface_spec_key(spec) -> tuple:
    """Hashable key for a surface spec, used to dedup mirror templates."""
    parts: list = []
    for s in _surface_list(spec):
        match s.type:
            case "aspheric":
                parts.append(("aspheric", s.curvature, s.conic, tuple(s.aspheric)))
            case "zernike":
                parts.append(("zernike", tuple(s.coeffs), s.r_norm))
    return tuple(parts)


def _asphere_surface_arrays(
    surface, n: int
) -> tuple[AsphericSurfaceGroup | None, ZernikeSurfaceGroup | None, Array]:
    """Return ``(asphere, zernike, offsets)`` for a group's surface.
    For a standalone Zernike surface (no aspheric base) the curvature / conic /
    aspheric default to a flat surface and the decenter is taken from the
    Zernike term.
    """
    asph, zern = _surface_components(surface)
    if asph is not None:
        offsets = asph.offsets
    elif zern is not None:
        offsets = zern.offsets
    else:  # pragma: no cover - _surface_components never returns (None, None)
        offsets = jnp.zeros((n, 2))
    return asph, zern, offsets


class _MirrorData(NamedTuple):
    """One mirror element's resolved fields, before the template/self-contained
    decision (:func:`mirrors_to_schemas`)."""

    group: OpticalElementGroup
    i: int
    asph_schema: AsphericSurfaceSchema | None
    zern_schema: ZernikeSurfaceSchema | None
    coating_schema: TabulatedCurveSchema | None
    offset: Array
    bsdf_schema: BSDFSchema | None
    reflectivity_scalar: float


def _mirror_base_key(d: _MirrorData) -> tuple | None:
    """Dedup key for a mirror's templatable fields (aspheric base + coating).

    ``None`` when the mirror has no aspheric base at all (a standalone
    Zernike surface); such mirrors never join a template. ``zernike`` and
    ``bsdf`` are deliberately excluded -- they stay per-mirror even when the
    aspheric base is shared (see :func:`mirrors_to_schemas`).
    """
    if d.asph_schema is None:
        return None
    return (_surface_spec_key(d.asph_schema), _curve_schema_to_key(d.coating_schema))


def mirrors_to_schemas(
    groups: list[OpticalElementGroup],
) -> tuple[dict[str, MirrorTemplateSchema], list[MirrorSchema]]:
    """Extract mirror schemas from OpticalElementGroup list.

    Each mirror is written as the joint of an optional template and its own
    fields, mirroring how loading resolves them (:func:`_resolve_surface`):

    * The *aspheric* base (curvature/conic/aspheric) plus ``coating`` are
      deduplicated into a shared template when two or more mirrors have the
      exact same combination. A mirror whose combination is unique to it (or
      has no aspheric base at all -- a standalone Zernike surface) gets no
      template: its curvature/conic/aspheric/coating are written directly.
    * ``zernike`` is always written directly on the mirror, never folded into
      a template, since it typically represents a per-panel measured figure
      error even when every panel shares the same base prescription.
    * ``bsdf`` is always per-mirror, as before.
    """
    data: list[_MirrorData] = []
    for group in groups:
        match group.interaction_module:
            case ReflectInteraction() as interaction:
                pass
            case _:
                continue

        coating_schema = _coating_to_curve_schema(interaction.reflectivity)
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
                    coating_schema=coating_schema,
                    offset=offsets[i],
                    bsdf_schema=_bsdf_to_schema(group.bsdf, i),
                    reflectivity_scalar=float(interaction.reflectivity_scalar[i]),
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
                    coating=d.coating_schema,
                )
            curvature = conic = aspheric = None
            coating = None
        else:
            template_name = None
            coating = d.coating_schema
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
                reflectivity=(
                    d.reflectivity_scalar if d.reflectivity_scalar != 1.0 else None
                ),
                coating=coating,
                id=f"M_{len(mirrors)}",
            )
        )

    return templates, mirrors


def _aperture_to_schema(
    aperture: Aperture, i: int
) -> CircularApertureSchema | PolygonApertureSchema:
    """Convert a domain aperture element to its schema representation."""
    match aperture:
        case DiskAperture(radii=radii, inner_radii=inner_radii):
            return CircularApertureSchema(
                radius=float(radii[i]),
                inner_radius=float(inner_radii[i]),
            )
        case PolygonAperture(vertices=vertices):
            return PolygonApertureSchema(
                vertices=[[float(v[0]), float(v[1])] for v in np.asarray(vertices[i])],
            )
        case _:
            raise ValueError(f"Unknown aperture type: {type(aperture)}")


def lenses_to_schemas(
    groups: list[OpticalElementGroup] | None,
) -> list[LensSchemaType]:
    """Extract lens schemas from OpticalElementGroup list; see ``_LENS_SPECS``."""
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
    coating_schema = _coating_to_curve_schema(interaction.transmittance)
    return AsphericDiskLensSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        surface=_surface_to_spec(asph, zern, i),
        n_inside=float(interaction.n_inside[i]),
        offset=_to_float_list(offsets[i]),
        transmittance=float(interaction.transmittance_scalar[i]),
        coating=coating_schema,
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
    coating_schema = _coating_to_curve_schema(interaction.transmittance)
    return PlanoSlabSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        thickness=float(interaction.thickness[i]),
        n_inside=float(interaction.n_inside[i]),
        transmittance=float(interaction.transmittance_scalar[i]),
        coating=coating_schema,
        stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def _extract_aspheric_disk_lenses(
    group: OpticalElementGroup, start: int
) -> list[AsphericDiskLensSchema]:
    """Extract every element of an aspheric-disk lens group, starting at index ``start``."""
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
    """Extract every element of a plano-slab group, starting at index ``start``."""
    interaction = group.interaction_module
    # _LENS_SPEC_BY_INTERACTION only ever routes here for a SlabInteraction
    # group; the assert both documents and narrows that for the type checker.
    assert isinstance(interaction, SlabInteraction)
    _, slab_zern = _surface_components(group.surface)
    if slab_zern is not None:
        raise ValueError("cannot serialise a Zernike figure error on a plano slab")
    return [_extract_plano_slab_lens(group, interaction, i, start + i) for i in range(len(group))]


class _LensSpec(NamedTuple):
    """Bidirectional spec for one lens type, mirroring ``_ObsSpec``.

    ``builder`` constructs one bucket's :class:`OpticalElementGroup` (see
    :func:`_build_lens_groups_by_stage`); ``extract_group`` is its inverse,
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


_SPEC_BY_GROUP: dict[type, _ObsSpec] = {spec.group: spec for spec in _OBSTRUCTION_SPECS}


def _extract_obstruction(spec: _ObsSpec, group: ObstructionGroup, i: int, counter: int):
    """Project element ``i`` of an obstruction group back to its schema."""
    kwargs: dict[str, object] = {"id": f"obs_{counter}"}
    for f in spec.fields:
        col = getattr(group, f.group_attr)
        if f.kind == "scalar":
            kwargs[f.schema_attr] = float(col[i])
        elif f.kind == "euler_matrix":
            kwargs[f.schema_attr] = _rotation_matrix_to_euler(np.asarray(col[i]))
        else:  # vec3
            kwargs[f.schema_attr] = _to_float_list(col[i])
    return spec.schema(**kwargs)


def obstructions_to_schemas(
    groups: list[ObstructionGroup] | None,
) -> list[ObstructionSchemaType]:
    """Extract obstruction schemas from an ObstructionGroup list.

    One schema per primitive, ``id``-numbered globally in traversal order.
    """
    if not groups:
        return []

    obstructions: list[ObstructionSchemaType] = []
    for group in groups:
        spec = _SPEC_BY_GROUP.get(type(group))
        if spec is None:
            raise ValueError(f"Unknown obstruction group type: {type(group)}")
        for i in range(len(group)):
            obstructions.append(_extract_obstruction(spec, group, i, len(obstructions)))
    return obstructions


def sensors_to_schemas(
    sensors: list[SensorGroup],
) -> list[SensorSchemaType]:
    """Extract sensor schemas from a SensorGroup list; see ``_SENSOR_SPECS``.

    One YAML entry per :class:`SensorGroup`: groups carrying multiple
    sensors are written with plural ``positions``/``orientations`` lists,
    so a multi-tile focal plane round-trips as a single group instead of
    being split into N single-tile groups.
    """
    result: list[SensorSchemaType] = []
    for counter, group in enumerate(sensors):
        spec = _SENSOR_SPEC_BY_GROUP.get(type(group))
        if spec is None:
            raise ValueError(f"Unknown sensor group type: {type(group)}")
        result.append(spec.extract(group, counter))
    return result


def _extract_square_group(
    group: SquareSensorGroup,
    counter: int,
) -> SquareSensorSchema:
    concentrator, gap, photodetector = _chain_to_schema_fields(group.chain)
    return SquareSensorSchema(
        positions=[_to_float_list(p) for p in group.positions],
        orientations=[_to_float_list(r) for r in group.rotations],
        width=group.width,
        height=group.height,
        bounds=list(group.bounds),
        edge_width=group.edge_width,
        concentrator=concentrator,
        gap=gap,
        photodetector=photodetector,
        id=f"sensor_{counter}",
    )


def _extract_hex_group(
    group: HexagonalSensorGroup,
    counter: int,
) -> HexagonalSensorSchema:
    hex_centers = np.asarray(group.hex_centers)
    concentrator, gap, photodetector = _chain_to_schema_fields(group.chain)
    return HexagonalSensorSchema(
        positions=[_to_float_list(p) for p in group.positions],
        orientations=[_to_float_list(r) for r in group.rotations],
        centers_x=_to_float_list(hex_centers[:, 0]),
        centers_y=_to_float_list(hex_centers[:, 1]),
        edge_width=group.edge_width,
        concentrator=concentrator,
        gap=gap,
        photodetector=photodetector,
        id=f"sensor_{counter}",
    )


class _SensorSpec(NamedTuple):
    """Bidirectional spec for one sensor-group type; see ``_ConcentratorSpec``."""

    type_name: str
    schema: type
    group: type
    build: Callable[..., SensorGroup]
    # Each entry's extract only ever accepts that entry's own SensorGroup
    # subclass (the driver looks it up by type(group) first); see _BsdfSpec.
    extract: Callable[..., SensorSchemaType]


# The single source of truth for sensor-group round-tripping; see
# _OBSTRUCTION_SPECS / _CONCENTRATOR_SPECS.
_SENSOR_SPECS: tuple[_SensorSpec, ...] = (
    _SensorSpec(
        "square", SquareSensorSchema, SquareSensorGroup, _build_square_group, _extract_square_group
    ),
    _SensorSpec(
        "hexagonal",
        HexagonalSensorSchema,
        HexagonalSensorGroup,
        _build_hex_group,
        _extract_hex_group,
    ),
)
_SENSOR_SPEC_BY_TYPE: dict[str, _SensorSpec] = {s.type_name: s for s in _SENSOR_SPECS}
_SENSOR_SPEC_BY_GROUP: dict[type, _SensorSpec] = {s.group: s for s in _SENSOR_SPECS}


class _ConcentratorSpec(NamedTuple):
    """Bidirectional spec for one concentrator type, mirroring ``_ObsSpec``.

    Unlike an obstruction, a concentrator's schema<->domain conversion isn't
    a flat field copy (truncation / wall-tilt reconstruction is involved), so
    each entry carries a pair of converter callables rather than a field
    table; the type-name-keyed table and type-agnostic load/save drivers
    below are otherwise the same idea.
    """

    type_name: str
    schema: type
    group: type
    # Each entry's converters only ever handle that entry's own concrete
    # domain / schema type (the drivers look them up by type(x) / x.type
    # first); see _BsdfSpec for why these aren't narrowly typed.
    to_schema: Callable[..., ConcentratorSchema]
    from_schema: Callable[..., Concentrator]


def _winston_to_schema(concentrator: WinstonCone) -> WinstonConeSchema:
    # entrance_apothem is the physical mouth at z=length; for a truncated
    # cone the depth reconstructs the wall on load. An untruncated cone is
    # written as length=None so reload is exact. "Full" <-> the mouth equals
    # the full-CPC mouth a2/s for the stored wall tilt s.
    ideal_mouth = concentrator.exit_apothem / concentrator.s
    truncated = not math.isclose(concentrator.entrance_apothem, ideal_mouth, rel_tol=1e-9)
    return WinstonConeSchema(
        n_sides=concentrator.n_sides,
        entrance_apothem=concentrator.entrance_apothem,
        exit_apothem=concentrator.exit_apothem,
        length=concentrator.length if truncated else None,
        reflectivity=concentrator.reflectivity,
        max_bounces=concentrator.max_bounces,
        orientation_deg=math.degrees(concentrator.orientation),
    )


def _winston_from_schema(schema: WinstonConeSchema) -> WinstonCone:
    return WinstonCone(
        n_sides=schema.n_sides,
        entrance_apothem=schema.entrance_apothem,
        exit_apothem=schema.exit_apothem,
        length=schema.length,
        reflectivity=schema.reflectivity,
        max_bounces=schema.max_bounces,
        orientation_deg=schema.orientation_deg,
    )


def _okumura_to_schema(concentrator: OkumuraCone) -> OkumuraConeSchema:
    # A None length reconstructs the Winston-equivalent depth on load;
    s = concentrator.exit_apothem / concentrator.entrance_apothem
    c = math.sqrt(1.0 - s * s)
    default_length = cpc_full_length(concentrator.exit_apothem, s, c)
    truncated = not math.isclose(concentrator.length, default_length, rel_tol=1e-9)
    return OkumuraConeSchema(
        n_sides=concentrator.n_sides,
        entrance_apothem=concentrator.entrance_apothem,
        exit_apothem=concentrator.exit_apothem,
        control_points=[[r, z] for r, z in concentrator.control_points],
        length=concentrator.length if truncated else None,
        reflectivity=concentrator.reflectivity,
        max_bounces=concentrator.max_bounces,
        orientation_deg=math.degrees(concentrator.orientation),
    )


def _okumura_from_schema(schema: OkumuraConeSchema) -> OkumuraCone:
    return OkumuraCone(
        n_sides=schema.n_sides,
        entrance_apothem=schema.entrance_apothem,
        exit_apothem=schema.exit_apothem,
        control_points=[(r, z) for r, z in schema.control_points],
        length=schema.length,
        reflectivity=schema.reflectivity,
        max_bounces=schema.max_bounces,
        orientation_deg=schema.orientation_deg,
    )


# The single source of truth for concentrator round-tripping, mirroring
# _OBSTRUCTION_SPECS. Adding a new concentrator is one entry here plus its
# schema (io.schemas) and domain (camera.optics) classes; the load/save
# drivers below are type-agnostic and raise for anything not registered
# (no silent drop-to-None on save).
_CONCENTRATOR_SPECS: tuple[_ConcentratorSpec, ...] = (
    _ConcentratorSpec(
        "winston", WinstonConeSchema, WinstonCone, _winston_to_schema, _winston_from_schema
    ),
    _ConcentratorSpec(
        "okumura", OkumuraConeSchema, OkumuraCone, _okumura_to_schema, _okumura_from_schema
    ),
)
_CONCENTRATOR_SPEC_BY_GROUP: dict[type, _ConcentratorSpec] = {
    s.group: s for s in _CONCENTRATOR_SPECS
}
_CONCENTRATOR_SPEC_BY_TYPE: dict[str, _ConcentratorSpec] = {
    s.type_name: s for s in _CONCENTRATOR_SPECS
}


def _concentrator_to_schema(
    concentrator: Concentrator | None,
) -> ConcentratorSchema | None:
    """Serialize a concentrator (``None`` -> ``None``); see ``_CONCENTRATOR_SPECS``.

    Raises for any :class:`~iactrace.camera.optics.concentrator.Concentrator`
    subclass without a registered spec, rather than silently dropping it --
    an unrepresentable concentrator is a large, silent physics change if
    saving just wrote "no concentrator" instead.
    """
    if concentrator is None:
        return None
    spec = _CONCENTRATOR_SPEC_BY_GROUP.get(type(concentrator))
    if spec is None:
        raise ValueError(
            f"{type(concentrator).__name__} is not representable in camera "
            "YAML; add a _ConcentratorSpec entry in iactrace.io.adapters "
            "plus a schema in iactrace.io.schemas."
        )
    return spec.to_schema(concentrator)


def _concentrator_from_schema(
    schema: ConcentratorSchema | None,
) -> Concentrator | None:
    """Rebuild a concentrator from its schema (``None`` -> no concentrator)."""
    if schema is None:
        return None
    spec = _CONCENTRATOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:
        raise ValueError(f"unknown concentrator schema: {type(schema).__name__}")
    return spec.from_schema(schema)


class _PhotoDetectorSpec(NamedTuple):
    """Bidirectional spec for one photodetector type; see ``_ConcentratorSpec``."""

    type_name: str
    schema: type
    group: type
    # Loosely typed for the same reason as _ConcentratorSpec: each entry's
    # converters only ever handle that entry's own concrete type.
    to_schema: Callable[..., PhotoDetectorSchema]
    from_schema: Callable[..., PhotoDetector]


def _constant_qe_to_schema(photodetector: ConstantQE) -> ConstantQESchema:
    return ConstantQESchema(qe=float(photodetector.qe))


def _constant_qe_from_schema(schema: ConstantQESchema) -> ConstantQE:
    return ConstantQE(schema.qe)


def _pmt_to_schema(photodetector: PMT) -> PMTSchema:
    # The photocathode figure round-trips through the exact same
    # (aspheric, zernike) decomposition mirrors and lenses use.
    asph, zern = _surface_components(photodetector.shape)
    return PMTSchema(
        qe=float(photodetector.qe),
        n_window=None if photodetector.n_window is None else float(photodetector.n_window),
        face_radius=float(photodetector.face_radius),
        surface=_surface_to_spec(asph, zern, 0),
        vertex_z=float(photodetector.vertex_z),
        # PMT resolves length=None to 2*face_radius at construction;
        # write the resolved value so the reload is exact.
        length=float(photodetector.length),
        n_facets=int(photodetector.n_facets),
    )


def _pmt_from_schema(schema: PMTSchema) -> PMT:
    return PMT(
        qe=schema.qe,
        n_window=schema.n_window,
        face_radius=schema.face_radius,
        surface=_single_element_surface(schema.surface),
        vertex_z=schema.vertex_z,
        length=schema.length,
        n_facets=schema.n_facets,
    )


# The single source of truth for photodetector round-tripping; see
# _CONCENTRATOR_SPECS.
_PHOTODETECTOR_SPECS: tuple[_PhotoDetectorSpec, ...] = (
    _PhotoDetectorSpec(
        "constant", ConstantQESchema, ConstantQE, _constant_qe_to_schema, _constant_qe_from_schema
    ),
    _PhotoDetectorSpec("pmt", PMTSchema, PMT, _pmt_to_schema, _pmt_from_schema),
)
_PHOTODETECTOR_SPEC_BY_GROUP: dict[type, _PhotoDetectorSpec] = {
    s.group: s for s in _PHOTODETECTOR_SPECS
}
_PHOTODETECTOR_SPEC_BY_TYPE: dict[str, _PhotoDetectorSpec] = {
    s.type_name: s for s in _PHOTODETECTOR_SPECS
}


def _photodetector_to_schema(photodetector: PhotoDetector) -> PhotoDetectorSchema:
    """Serialize a photodetector; see ``_PHOTODETECTOR_SPECS``.

    Raises for any :class:`~iactrace.camera.detector.photodetector.PhotoDetector`
    subclass without a registered spec, rather than silently falling back to
    a perfect ``ConstantQE(1.0)`` -- that fallback would be a large, silent
    change to the detection efficiency.
    """
    spec = _PHOTODETECTOR_SPEC_BY_GROUP.get(type(photodetector))
    if spec is None:
        raise ValueError(
            f"{type(photodetector).__name__} is not representable in camera "
            "YAML; add a _PhotoDetectorSpec entry in iactrace.io.adapters "
            "plus a schema in iactrace.io.schemas."
        )
    return spec.to_schema(photodetector)


def _photodetector_from_schema(schema: PhotoDetectorSchema | None) -> PhotoDetector:
    """Rebuild a photodetector from its schema (``None`` -> ``ConstantQE(1.0)``)."""
    if schema is None:
        return ConstantQE(1.0)
    spec = _PHOTODETECTOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:
        raise ValueError(f"unknown photodetector schema: {type(schema).__name__}")
    return spec.from_schema(schema)


def _chain_to_schema_fields(
    chain: DetectionChain,
) -> tuple[ConcentratorSchema | None, float, PhotoDetectorSchema | None]:
    """Project a detection chain to its ``(concentrator, gap, photodetector)`` schema.

    Photodetectors (:class:`~iactrace.camera.detector.photodetector.ConstantQE` /
    :class:`~iactrace.camera.detector.pmt.PMT`) and concentrators
    (:class:`~iactrace.camera.optics.winston.WinstonCone` /
    :class:`~iactrace.camera.optics.okumura.OkumuraCone`) round-trip exactly;
    any other subclass raises (see ``_photodetector_to_schema`` /
    ``_concentrator_to_schema``) rather than silently degrading the physics.
    The trivial perfect-QE photodetector is emitted as ``None`` so a
    geometry-only sensor group serializes without a redundant
    ``photodetector:`` block.
    """
    concentrator = _concentrator_to_schema(chain.concentrator)
    photodetector: PhotoDetectorSchema | None
    if isinstance(chain.photodetector, ConstantQE) and float(chain.photodetector.qe) == 1.0:
        photodetector = None
    else:
        photodetector = _photodetector_to_schema(chain.photodetector)
    return concentrator, float(chain.gap), photodetector


def telescope_to_schema(telescope: Telescope) -> TelescopeConfigSchema:
    """Convert a Telescope to a TelescopeConfigSchema (telescope-only file).

    The output describes mirrors/lenses/obstructions plus the camera frame
    (``camera_position`` / ``camera_rotation``). Detector geometry lives in
    a separate camera YAML; use :func:`camera_to_file_schema` for that.
    """
    templates, mirror_schemas = mirrors_to_schemas(telescope.mirror_groups)

    cam_pos = np.asarray(telescope.camera_position)
    cam_rot = np.asarray(telescope.camera_rotation)

    return TelescopeConfigSchema(
        telescope=TelescopeMetadataSchema(
            name=telescope.name,
            camera_position=_to_float_list(cam_pos),
            camera_rotation=_to_float_list(cam_rot),
        ),
        mirror_templates=templates,
        mirrors=mirror_schemas,
        lenses=lenses_to_schemas(telescope.lens_groups),
        obstructions=obstructions_to_schemas(telescope.obstruction_groups),
    )


def camera_to_file_schema(camera: Camera) -> CameraFileSchema:
    """Convert a Camera to a standalone CameraFileSchema.

    Sensor positions are written in the camera-local frame.
    """
    sensor_schemas: list[SensorSchemaType] = []
    if camera.sensor_groups:
        sensor_schemas = sensors_to_schemas(camera.sensor_groups)

    return CameraFileSchema(sensors=sensor_schemas)