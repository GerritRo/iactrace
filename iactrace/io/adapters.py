from __future__ import annotations

from collections import defaultdict
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

# Type aliases for discriminated union schema types
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


# ---------- Private helpers ----------


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
        disk_mirrors = [m for m in stage_mirrors if isinstance(m.aperture, CircularApertureSchema)]
        poly_mirrors = [m for m in stage_mirrors if isinstance(m.aperture, PolygonApertureSchema)]

        if disk_mirrors:
            aperture = DiskAperture(
                radii=jnp.asarray([m.aperture.radius for m in disk_mirrors]),
                inner_radii=jnp.asarray([m.aperture.inner_radius for m in disk_mirrors]),
            )
            key, subkey = jax.random.split(key)
            groups.append(_build_mirror_group(disk_mirrors, aperture, stage, n_samples, sample_key=subkey))

        if poly_mirrors:
            by_nverts: dict[int, list[_ParsedMirror]] = defaultdict(list)
            for m in poly_mirrors:
                n_verts = len(m.aperture.vertices)
                by_nverts[n_verts].append(m)

            for _n_verts, mirror_list in by_nverts.items():
                vertices_list = []
                for m in mirror_list:
                    verts = jnp.asarray(m.aperture.vertices)
                    vertices_list.append(_ensure_ccw(verts))
                vertices = jnp.stack(vertices_list)
                aperture = PolygonAperture(vertices=vertices, n_vertices=int(vertices.shape[1]))
                key, subkey = jax.random.split(key)
                groups.append(_build_mirror_group(mirror_list, aperture, stage, n_samples, sample_key=subkey))

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

    Groups by type and optical_stage, constructs OpticalElementGroup directly.
    """
    if not lenses:
        return []

    groups: list[OpticalElementGroup] = []

    # Group by (type, optical_stage)
    by_key: dict[tuple[str, int], list[LensSchemaType]] = defaultdict(list)
    for lens in lenses:
        by_key[(lens.type, lens.optical_stage)].append(lens)

    _builders = {
        "aspheric_disk": _build_aspheric_disk_lens_group,
        "plano_slab": _build_plano_slab_group,
    }

    for (ltype, stage), lens_list in by_key.items():
        builder = _builders[ltype]
        key, subkey = jax.random.split(key)
        groups.append(builder(lens_list, stage, sample_key=subkey))

    return groups


def _build_aspheric_disk_lens_group(
    lenses: list[LensSchemaType], stage: int, *, sample_key: Array
) -> OpticalElementGroup:
    """Build an aspheric-disk refractive group via the telescope helper.

    Delegates to :func:`iactrace.telescope.lenses.refractive_group` once
    schema fields have been projected into arrays.
    """
    from ..telescope.lenses import refractive_group

    n_elements = len(lenses)
    aperture = DiskAperture(
        radii=jnp.asarray([lens.radius for lens in lenses]),
        inner_radii=jnp.zeros(n_elements),
    )

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
    lenses: list[LensSchemaType], stage: int, *, sample_key: Array
) -> OpticalElementGroup:
    """Build a plano-slab group via the telescope helper.

    Delegates to :func:`iactrace.telescope.lenses.slab_group` once schema
    fields have been projected into arrays.
    """
    from ..telescope.lenses import slab_group

    n_elements = len(lenses)
    aperture = DiskAperture(
        radii=jnp.asarray([lens.radius for lens in lenses]),
        inner_radii=jnp.zeros(n_elements),
    )

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
    if not obstructions:
        return []

    by_type: dict[str, list[ObstructionSchemaType]] = defaultdict(list)
    for obs in obstructions:
        by_type[obs.type].append(obs)

    groups: list[ObstructionGroup] = []
    _builders = {
        "cylinder": _build_cylinder_group,
        "open_cylinder": _build_open_cylinder_group,
        "box": _build_box_group,
        "sphere": _build_sphere_group,
        "oriented_box": _build_oriented_box_group,
        "triangle": _build_triangle_group,
    }

    for otype, schemas in by_type.items():
        builder = _builders[otype]
        groups.append(builder(schemas))

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
    positions = [list(p) for p in schema.position_list]
    rotations = [list(r) for r in schema.orientation_list]

    if isinstance(schema, SquareSensorSchema):
        bounds = schema.bounds
        return SquareSensorGroup(
            positions=positions,
            rotations=rotations,
            width=schema.width,
            height=schema.height,
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            edge_width=schema.edge_width,
        )
    elif isinstance(schema, HexagonalSensorSchema):
        hex_centers = [
            [x, y] for x, y in zip(schema.centers_x, schema.centers_y, strict=False)
        ]
        return HexagonalSensorGroup(
            positions=positions,
            rotations=rotations,
            hex_centers=hex_centers,
            edge_width=schema.edge_width,
        )
    raise ValueError(f"Unknown sensor schema type: {type(schema)}")


# ---------- Domain -> Schema (saving) ----------


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

            mirror_kwargs: dict[str, object] = {
                "position": position,
                "orientation": orientation,
                "aperture": aperture,
                "template": template_name,
                "id": f"M_{len(mirrors)}",
            }

            if group.optical_stage != 0:
                mirror_kwargs["stage"] = group.optical_stage

            offset = _to_float_list(group.surface.offsets[i])
            if not (offset[0] == 0.0 and offset[1] == 0.0):
                mirror_kwargs["offset"] = offset

            if bsdf_scale_val != 0.0:
                mirror_kwargs["bsdf_scale"] = bsdf_scale_val

            mirrors.append(MirrorSchema(**mirror_kwargs))

    return templates, mirrors


def _aperture_to_schema(
    aperture: Aperture, i: int
) -> CircularApertureSchema | PolygonApertureSchema:
    """Convert a domain aperture element to its schema representation."""
    if isinstance(aperture, DiskAperture):
        return CircularApertureSchema(
            radius=float(aperture.radii[i]),
            inner_radius=float(aperture.inner_radii[i]),
        )
    elif isinstance(aperture, PolygonAperture):
        verts = [[float(v[0]), float(v[1])] for v in np.asarray(aperture.vertices[i])]
        return PolygonApertureSchema(vertices=verts)
    raise ValueError(f"Unknown aperture type: {type(aperture)}")


def lenses_to_schemas(
    groups: list[OpticalElementGroup] | None,
) -> list[LensSchemaType]:
    """Extract lens schemas from OpticalElementGroup list."""
    if not groups:
        return []

    lenses: list[LensSchemaType] = []

    _extractors: dict[type, object] = {
        RefractInteraction: _extract_aspheric_disk_lens,
        SlabInteraction: _extract_plano_slab_lens,
    }

    for group in groups:
        extractor = _extractors.get(type(group.interaction_module))
        if extractor is None:
            continue
        for i in range(len(group)):
            lenses.append(extractor(group, i, len(lenses)))

    return lenses


def _extract_aspheric_disk_lens(
    group: OpticalElementGroup, i: int, counter: int
) -> AsphericDiskLensSchema:
    """Extract an AsphericDiskLensSchema from element i of a group."""
    aspheric_raw = _strip_trailing_zeros(_to_float_list(group.surface.aspherics[i]))
    return AsphericDiskLensSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        curvature=float(group.surface.curvatures[i]),
        conic=float(group.surface.conics[i]),
        radius=float(group.aperture.radii[i]),
        n_inside=float(group.interaction_module.n_inside[i]),
        n_outside=float(group.interaction_module.n_outside),
        aspheric=aspheric_raw,
        offset=_to_float_list(group.surface.offsets[i]),
        transmittance=float(group.interaction_module.transmittance[i]),
        optical_stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def _extract_plano_slab_lens(
    group: OpticalElementGroup, i: int, counter: int
) -> PlanoSlabSchema:
    """Extract a PlanoSlabSchema from element i of a group."""
    return PlanoSlabSchema(
        position=_to_float_list(group.positions[i]),
        orientation=_to_float_list(group.rotations[i]),
        radius=float(group.aperture.radii[i]),
        thickness=float(group.interaction_module.thickness[i]),
        n_inside=float(group.interaction_module.n_inside[i]),
        n_outside=float(group.interaction_module.n_outside),
        transmittance=float(group.interaction_module.transmittance[i]),
        optical_stage=group.optical_stage,
        id=f"lens_{counter}",
    )


def obstructions_to_schemas(
    groups: list[ObstructionGroup] | None,
) -> list[ObstructionSchemaType]:
    """Extract obstruction schemas from ObstructionGroup list."""
    if not groups:
        return []

    _extractors: dict[type, object] = {
        CylinderGroup: _extract_cylinder,
        OpenCylinderGroup: _extract_open_cylinder,
        BoxGroup: _extract_box,
        SphereGroup: _extract_sphere,
        OrientedBoxGroup: _extract_oriented_box,
        TriangleGroup: _extract_triangle,
    }

    obstructions: list[ObstructionSchemaType] = []
    counter = 0

    for group in groups:
        extractor = _extractors[type(group)]
        for i in range(len(group)):
            obstructions.append(extractor(group, i, counter))
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
    _extractors: dict[type, object] = {
        SquareSensorGroup: _extract_square_group,
        HexagonalSensorGroup: _extract_hex_group,
    }

    result: list[SensorSchemaType] = []
    for counter, group in enumerate(sensors):
        extractor = _extractors[type(group)]
        result.append(extractor(group, counter))

    return result


def _placement_kwargs(group: SensorGroup) -> dict[str, object]:
    """Singular ``position``/``orientation`` for N=1, plural lists otherwise."""
    positions = [_to_float_list(p) for p in group.positions]
    rotations = [_to_float_list(r) for r in group.rotations]
    if len(positions) == 1:
        return {"position": positions[0], "orientation": rotations[0]}
    return {"positions": positions, "orientations": rotations}


def _extract_square_group(
    group: SquareSensorGroup, counter: int,
) -> SquareSensorSchema:
    schema_kwargs: dict[str, object] = {
        **_placement_kwargs(group),
        "width": group.width,
        "height": group.height,
        "bounds": list(group.bounds),
        "id": f"sensor_{counter}",
    }
    if group.edge_width > 0:
        schema_kwargs["edge_width"] = group.edge_width
    return SquareSensorSchema(**schema_kwargs)


def _extract_hex_group(
    group: HexagonalSensorGroup, counter: int,
) -> HexagonalSensorSchema:
    hex_centers = np.asarray(group.hex_centers)
    schema_kwargs: dict[str, object] = {
        **_placement_kwargs(group),
        "centers_x": _to_float_list(hex_centers[:, 0]),
        "centers_y": _to_float_list(hex_centers[:, 1]),
        "id": f"sensor_{counter}",
    }
    if group.edge_width > 0:
        schema_kwargs["edge_width"] = group.edge_width
    return HexagonalSensorSchema(**schema_kwargs)


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