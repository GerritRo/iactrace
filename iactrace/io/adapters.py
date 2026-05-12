from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..camera.layout import HexagonalSensorGroup, SquareSensorGroup
from ..core.apertures import Aperture, DiskAperture, PolygonAperture
from ..core.bsdf import GaussianBSDF
from ..core.interactions import (
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
from ..core.transforms import euler_to_matrix
from .schemas import (
    AsphericDiskLensSchema,
    BoxObstructionSchema,
    BSDFSchema,
    CameraConfigSchema,
    CameraFileSchema,
    CircularApertureSchema,
    CylinderObstructionSchema,
    HexagonalSensorSchema,
    MirrorSchema,
    MirrorTemplateSchema,
    OpenCylinderObstructionSchema,
    OrientedBoxObstructionSchema,
    PlanoSlabSchema,
    PolygonApertureSchema,
    SphereObstructionSchema,
    SquareSensorSchema,
    SurfaceSchema,
    TelescopeConfigSchema,
    TelescopeMetadataSchema,
    TriangleObstructionSchema,
)

if TYPE_CHECKING:
    from ..camera import Camera
    from ..camera.layout import SensorGroup
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
    offset: list[float]
    stage: int
    aperture: CircularApertureSchema | PolygonApertureSchema
    bsdf_scale: float


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


def _resolve_surface(
    mirror: MirrorSchema, template: MirrorTemplateSchema
) -> tuple[float, float, list[float]]:
    """Resolve surface parameters from mirror + template (mirror overrides template)."""
    surface = template.surface
    curvature = mirror.curvature if mirror.curvature is not None else surface.curvature
    conic = mirror.conic if mirror.conic is not None else surface.conic
    aspheric = mirror.aspheric if mirror.aspheric is not None else surface.aspheric
    return curvature, conic, aspheric


def _resolve_bsdf_scale(mirror: MirrorSchema, template: MirrorTemplateSchema) -> float:
    """Resolve BSDF scale from mirror + template."""
    if mirror.bsdf_scale is not None:
        return mirror.bsdf_scale
    if template.bsdf is not None:
        return template.bsdf.scale
    return 0.0


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
        template = templates[mirror.template]
        curvature, conic, aspheric = _resolve_surface(mirror, template)
        bsdf_scale = _resolve_bsdf_scale(mirror, template)

        parsed.append(_ParsedMirror(
            position=mirror.position,
            orientation=mirror.orientation,
            curvature=curvature,
            conic=conic,
            aspheric=aspheric,
            offset=mirror.offset,
            stage=mirror.stage,
            aperture=mirror.aperture,
            bsdf_scale=bsdf_scale,
        ))

    groups: list[OpticalElementGroup] = []

    # Group by stage
    by_stage: dict[int, list[_ParsedMirror]] = defaultdict(list)
    for p in parsed:
        by_stage[p.stage].append(p)

    for stage, stage_mirrors in sorted(by_stage.items()):
        for bucket in _bucket_by_aperture_signature(stage_mirrors, lambda m: m.aperture):
            aperture = _aperture_from_schemas([m.aperture for m in bucket])
            key, subkey = jax.random.split(key)
            groups.append(_build_mirror_group(bucket, aperture, stage, n_samples, sample_key=subkey))

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
    to :func:`iactrace.telescope.mirrors.mirror_group`, the canonical
    reflective-group builder for the whole project.
    """
    from ..telescope.mirrors import mirror_group

    n_elements = len(mirrors)

    scales = jnp.asarray([m.bsdf_scale for m in mirrors])
    bsdf = None if bool(jnp.all(scales == 0)) else GaussianBSDF(scale=scales)

    return mirror_group(
        positions=jnp.asarray([m.position for m in mirrors]),
        rotations=jnp.asarray([m.orientation for m in mirrors]),
        curvatures=jnp.asarray([m.curvature for m in mirrors]),
        conics=jnp.asarray([m.conic for m in mirrors]),
        aspherics=_pad_aspherics([m.aspheric for m in mirrors]),
        offsets=jnp.asarray([m.offset for m in mirrors]),
        aperture=aperture,
        reflectivity=jnp.ones(n_elements),
        bsdf=bsdf,
        sample_key=sample_key,
        optical_stage=stage,
        n_samples=n_samples,
    )


def lenses_from_schemas(
    lenses: list[LensSchemaType],
    *,
    key: Array,
) -> list[OpticalElementGroup]:
    """Convert validated lens schemas to OpticalElementGroup domain objects.

    Groups by ``(type, stage, aperture_signature)`` — mirroring how
    :func:`mirrors_from_schemas` groups mirrors — and constructs one
    :class:`OpticalElementGroup` per bucket.
    """
    aspheric_disks: list[AsphericDiskLensSchema] = []
    plano_slabs: list[PlanoSlabSchema] = []
    for lens in lenses:
        match lens.type:
            case "aspheric_disk":
                aspheric_disks.append(lens)
            case "plano_slab":
                plano_slabs.append(lens)

    groups: list[OpticalElementGroup] = []
    key, groups = _build_lens_groups_by_stage(aspheric_disks, _build_aspheric_disk_lens_group, key, groups)
    key, groups = _build_lens_groups_by_stage(plano_slabs, _build_plano_slab_group, key, groups)
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

    return refractive_group(
        positions=jnp.asarray([lens.position for lens in lenses]),
        rotations=jnp.asarray([lens.orientation for lens in lenses]),
        curvatures=jnp.asarray([lens.curvature for lens in lenses]),
        conics=jnp.asarray([lens.conic for lens in lenses]),
        aspherics=_pad_aspherics([lens.aspheric for lens in lenses]),
        offsets=jnp.asarray([lens.offset for lens in lenses]),
        aperture=aperture,
        n_inside=jnp.asarray([lens.n_inside for lens in lenses]),
        n_outside=float(lenses[0].n_outside),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        sample_key=sample_key,
        optical_stage=stage,
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

    return slab_group(
        positions=jnp.asarray([lens.position for lens in lenses]),
        rotations=jnp.asarray([lens.orientation for lens in lenses]),
        aperture=aperture,
        n_inside=jnp.asarray([lens.n_inside for lens in lenses]),
        n_outside=float(lenses[0].n_outside),
        thickness=jnp.asarray([lens.thickness for lens in lenses]),
        transmittance=jnp.asarray([lens.transmittance for lens in lenses]),
        sample_key=sample_key,
        optical_stage=stage,
    )


def obstructions_from_schemas(
    obstructions: list[ObstructionSchemaType],
) -> list[ObstructionGroup]:
    """Convert validated obstruction schemas to ObstructionGroup domain objects."""
    cylinders: list[CylinderObstructionSchema] = []
    open_cyls: list[OpenCylinderObstructionSchema] = []
    boxes: list[BoxObstructionSchema] = []
    spheres: list[SphereObstructionSchema] = []
    ori_boxes: list[OrientedBoxObstructionSchema] = []
    triangles: list[TriangleObstructionSchema] = []

    for obs in obstructions:
        match obs.type:
            case "cylinder":
                cylinders.append(obs)
            case "open_cylinder":
                open_cyls.append(obs)
            case "box":
                boxes.append(obs)
            case "sphere":
                spheres.append(obs)
            case "oriented_box":
                ori_boxes.append(obs)
            case "triangle":
                triangles.append(obs)

    groups: list[ObstructionGroup] = []
    if cylinders:
        groups.append(_build_cylinder_group(cylinders))
    if open_cyls:
        groups.append(_build_open_cylinder_group(open_cyls))
    if boxes:
        groups.append(_build_box_group(boxes))
    if spheres:
        groups.append(_build_sphere_group(spheres))
    if ori_boxes:
        groups.append(_build_oriented_box_group(ori_boxes))
    if triangles:
        groups.append(_build_triangle_group(triangles))
    return groups


def _build_cylinder_group(schemas: list[CylinderObstructionSchema]) -> CylinderGroup:
    return CylinderGroup(
        p1=[s.p1 for s in schemas],
        p2=[s.p2 for s in schemas],
        r=[s.r for s in schemas],
    )


def _build_open_cylinder_group(schemas: list[OpenCylinderObstructionSchema]) -> OpenCylinderGroup:
    return OpenCylinderGroup(
        p1=[s.p1 for s in schemas],
        p2=[s.p2 for s in schemas],
        r=[s.r for s in schemas],
    )


def _build_box_group(schemas: list[BoxObstructionSchema]) -> BoxGroup:
    return BoxGroup(
        p1=[s.p1 for s in schemas],
        p2=[s.p2 for s in schemas],
    )


def _build_sphere_group(schemas: list[SphereObstructionSchema]) -> SphereGroup:
    return SphereGroup(
        centers=[s.center for s in schemas],
        radii=[s.r for s in schemas],
    )


def _build_oriented_box_group(schemas: list[OrientedBoxObstructionSchema]) -> OrientedBoxGroup:
    rotations = []
    for s in schemas:
        euler = jnp.asarray(s.rotation)
        rot_matrix = euler_to_matrix(euler)
        rotations.append(rot_matrix)
    return OrientedBoxGroup(
        centers=[s.center for s in schemas],
        half_extents=[s.half_extents for s in schemas],
        rotations=jnp.stack(rotations),
    )


def _build_triangle_group(schemas: list[TriangleObstructionSchema]) -> TriangleGroup:
    return TriangleGroup(
        v0=[s.v0 for s in schemas],
        v1=[s.v1 for s in schemas],
        v2=[s.v2 for s in schemas],
    )


def sensor_from_schema(
    schema: SquareSensorSchema | HexagonalSensorSchema,
) -> SensorGroup:
    """Convert a validated sensor schema to a SensorGroup domain object.

    The schema ``position`` / ``orientation`` are interpreted as
    **camera-local** coordinates — the camera file format is the single
    source of truth and always speaks in the camera's own frame.
    """
    positions = [list(p) for p in schema.positions]
    rotations = [list(r) for r in schema.orientations]

    match schema.type:
        case "square":
            b = schema.bounds
            return SquareSensorGroup(
                positions=positions,
                rotations=rotations,
                width=schema.width,
                height=schema.height,
                bounds=(b[0], b[1], b[2], b[3]),
                edge_width=schema.edge_width,
            )
        case "hexagonal":
            return HexagonalSensorGroup(
                positions=positions,
                rotations=rotations,
                hex_centers=[
                    [x, y] for x, y in zip(schema.centers_x, schema.centers_y, strict=False)
                ],
                edge_width=schema.edge_width,
            )


# Domain -> Schema (saving)


def mirrors_to_schemas(
    groups: list[OpticalElementGroup],
) -> tuple[dict[str, MirrorTemplateSchema], list[MirrorSchema]]:
    """Extract mirror schemas from OpticalElementGroup list.

    Returns templates dict + mirror list. Deduplicates surface params into templates.
    BSDF is stored per-mirror (not part of the dedup key) since mirrors sharing a
    surface template can have different BSDF scales.
    """
    templates: dict[str, MirrorTemplateSchema] = {}
    mirrors: list[MirrorSchema] = []
    template_counter = 0
    surface_to_template: dict[tuple, str] = {}

    for group in groups:
        bsdf_scales = None
        if isinstance(group.bsdf, GaussianBSDF):
            bsdf_scales = group.bsdf.scale

        for i in range(len(group)):
            curvature = float(group.surface.curvatures[i])
            conic = float(group.surface.conics[i])
            aspheric_raw = _strip_trailing_zeros(_to_float_list(group.surface.aspherics[i]))

            bsdf_scale_val = float(bsdf_scales[i]) if bsdf_scales is not None else 0.0
            surface_key = (curvature, conic, tuple(aspheric_raw))

            if surface_key not in surface_to_template:
                template_name = f"template_{template_counter}"
                template_counter += 1
                surface_to_template[surface_key] = template_name

                bsdf_schema = None
                if bsdf_scale_val != 0.0:
                    bsdf_schema = BSDFSchema(scale=bsdf_scale_val)

                templates[template_name] = MirrorTemplateSchema(
                    surface=SurfaceSchema(
                        curvature=curvature,
                        conic=conic,
                        aspheric=aspheric_raw if aspheric_raw else [],
                    ),
                    bsdf=bsdf_schema,
                )

            template_name = surface_to_template[surface_key]
            position = _to_float_list(group.positions[i])
            orientation = _to_float_list(group.rotations[i])

            # Build aperture schema
            aperture = _aperture_to_schema(group.aperture, i)

            offset = _to_float_list(group.surface.offsets[i])
            mirrors.append(MirrorSchema(
                position=position,
                orientation=orientation,
                aperture=aperture,
                template=template_name,
                stage=group.optical_stage,
                offset=offset,
                bsdf_scale=bsdf_scale_val if bsdf_scale_val != 0.0 else None,
                id=f"M_{len(mirrors)}",
            ))

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
    """Extract lens schemas from OpticalElementGroup list."""
    if not groups:
        return []

    lenses: list[LensSchemaType] = []
    for group in groups:
        match group.interaction_module:
            case RefractInteraction() as interaction:
                for i in range(len(group)):
                    lenses.append(_extract_aspheric_disk_lens(group, interaction, i, len(lenses)))
            case SlabInteraction() as interaction:
                for i in range(len(group)):
                    lenses.append(_extract_plano_slab_lens(group, interaction, i, len(lenses)))
            case _:
                continue
    return lenses


def _extract_aspheric_disk_lens(
    group: OpticalElementGroup,
    interaction: RefractInteraction,
    i: int,
    counter: int,
) -> AsphericDiskLensSchema:
    """Extract an AsphericDiskLensSchema from element i of a group."""
    aspheric_raw = _strip_trailing_zeros(_to_float_list(group.surface.aspherics[i]))
    return AsphericDiskLensSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        curvature=float(group.surface.curvatures[i]),
        conic=float(group.surface.conics[i]),
        n_inside=float(interaction.n_inside[i]),
        n_outside=float(interaction.n_outside),
        aspheric=aspheric_raw,
        offset=_to_float_list(group.surface.offsets[i]),
        transmittance=float(interaction.transmittance[i]),
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
    return PlanoSlabSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        aperture=_aperture_to_schema(group.aperture, i),
        thickness=float(interaction.thickness[i]),
        n_inside=float(interaction.n_inside[i]),
        n_outside=float(interaction.n_outside),
        transmittance=float(interaction.transmittance[i]),
        stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def obstructions_to_schemas(
    groups: list[ObstructionGroup] | None,
) -> list[ObstructionSchemaType]:
    """Extract obstruction schemas from ObstructionGroup list."""
    if not groups:
        return []

    obstructions: list[ObstructionSchemaType] = []
    counter = 0
    for group in groups:
        for i in range(len(group)):
            match group:
                case CylinderGroup():
                    obstructions.append(_extract_cylinder(group, i, counter))
                case OpenCylinderGroup():
                    obstructions.append(_extract_open_cylinder(group, i, counter))
                case BoxGroup():
                    obstructions.append(_extract_box(group, i, counter))
                case SphereGroup():
                    obstructions.append(_extract_sphere(group, i, counter))
                case OrientedBoxGroup():
                    obstructions.append(_extract_oriented_box(group, i, counter))
                case TriangleGroup():
                    obstructions.append(_extract_triangle(group, i, counter))
                case _:
                    raise ValueError(f"Unknown obstruction group type: {type(group)}")
            counter += 1
    return obstructions


def _extract_cylinder(group: CylinderGroup, i: int, counter: int) -> CylinderObstructionSchema:
    return CylinderObstructionSchema(
        p1=_to_float_list(group.p1[i]),
        p2=_to_float_list(group.p2[i]),
        r=float(group.r[i]),
        id=f"obs_{counter}",
    )


def _extract_open_cylinder(group: OpenCylinderGroup, i: int, counter: int) -> OpenCylinderObstructionSchema:
    return OpenCylinderObstructionSchema(
        p1=_to_float_list(group.p1[i]),
        p2=_to_float_list(group.p2[i]),
        r=float(group.r[i]),
        id=f"obs_{counter}",
    )


def _extract_box(group: BoxGroup, i: int, counter: int) -> BoxObstructionSchema:
    return BoxObstructionSchema(
        p1=_to_float_list(group.p1[i]),
        p2=_to_float_list(group.p2[i]),
        id=f"obs_{counter}",
    )


def _extract_sphere(group: SphereGroup, i: int, counter: int) -> SphereObstructionSchema:
    return SphereObstructionSchema(
        center=_to_float_list(group.centers[i]),
        r=float(group.radii[i]),
        id=f"obs_{counter}",
    )


def _extract_oriented_box(group: OrientedBoxGroup, i: int, counter: int) -> OrientedBoxObstructionSchema:
    rotation_matrix = np.asarray(group.rotations[i])
    euler = _rotation_matrix_to_euler(rotation_matrix)
    return OrientedBoxObstructionSchema(
        center=_to_float_list(group.centers[i]),
        half_extents=_to_float_list(group.half_extents[i]),
        rotation=euler,
        id=f"obs_{counter}",
    )


def _extract_triangle(group: TriangleGroup, i: int, counter: int) -> TriangleObstructionSchema:
    return TriangleObstructionSchema(
        v0=_to_float_list(group.v0[i]),
        v1=_to_float_list(group.v1[i]),
        v2=_to_float_list(group.v2[i]),
        id=f"obs_{counter}",
    )


def sensors_to_schemas(
    sensors: list[SensorGroup],
) -> list[SensorSchemaType]:
    """Extract sensor schemas from a SensorGroup list.

    One YAML entry per :class:`SensorGroup`: groups carrying multiple
    sensors are written with plural ``positions``/``orientations`` lists,
    so a multi-tile focal plane round-trips as a single group instead of
    being split into N single-tile groups.
    """
    result: list[SensorSchemaType] = []
    for counter, group in enumerate(sensors):
        match group:
            case SquareSensorGroup():
                result.append(_extract_square_group(group, counter))
            case HexagonalSensorGroup():
                result.append(_extract_hex_group(group, counter))
            case _:
                raise ValueError(f"Unknown sensor group type: {type(group)}")
    return result


def _extract_square_group(
    group: SquareSensorGroup, counter: int,
) -> SquareSensorSchema:
    return SquareSensorSchema(
        positions=[_to_float_list(p) for p in group.positions],
        orientations=[_to_float_list(r) for r in group.rotations],
        width=group.width,
        height=group.height,
        bounds=list(group.bounds),
        edge_width=group.edge_width,
        id=f"sensor_{counter}",
    )


def _extract_hex_group(
    group: HexagonalSensorGroup, counter: int,
) -> HexagonalSensorSchema:
    hex_centers = np.asarray(group.hex_centers)
    return HexagonalSensorSchema(
        positions=[_to_float_list(p) for p in group.positions],
        orientations=[_to_float_list(r) for r in group.rotations],
        centers_x=_to_float_list(hex_centers[:, 0]),
        centers_y=_to_float_list(hex_centers[:, 1]),
        edge_width=group.edge_width,
        id=f"sensor_{counter}",
    )


def camera_to_schema(camera: Camera) -> CameraConfigSchema:
    """Convert a Camera to its schema representation.

    Only :class:`~iactrace.camera.photosensor.UniformQE` photosensors
    round-trip exactly. Any other :class:`~iactrace.camera.photosensor.PhotoSensor`
    subclass is not yet representable in the camera YAML; ``camera_to_schema``
    emits a :class:`UserWarning` and falls back to ``quantum_efficiency=1.0``
    so that ``Camera.to_yaml()`` does not crash mid-save.
    """
    return CameraConfigSchema()


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

    Sensor positions are written in the camera-local frame — that is the
    only frame the camera file format knows about.
    """
    sensor_schemas: list[SensorSchemaType] = []
    if camera.sensor_groups:
        sensor_schemas = sensors_to_schemas(camera.sensor_groups)

    return CameraFileSchema(
        camera=camera_to_schema(camera),
        sensors=sensor_schemas,
    )
