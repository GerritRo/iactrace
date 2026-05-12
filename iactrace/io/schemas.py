from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

# Reusable constrained types

Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Bounds4 = Annotated[list[float], Field(min_length=4, max_length=4)]


# Aperture schemas


class CircularApertureSchema(BaseModel):
    type: Literal["circular"] = "circular"
    radius: float = Field(gt=0)
    inner_radius: float = Field(ge=0, default=0.0)


class PolygonApertureSchema(BaseModel):
    type: Literal["polygon"] = "polygon"
    vertices: list[Vec2] = Field(min_length=3)


ApertureSchema = Annotated[
    CircularApertureSchema | PolygonApertureSchema,
    Field(discriminator="type"),
]


# Mirror template & mirror schemas


class SurfaceSchema(BaseModel):
    curvature: float
    conic: float
    aspheric: list[float] = Field(default_factory=list)


class BSDFSchema(BaseModel):
    type: str = "gaussian"
    scale: float = Field(ge=0, default=0.0)


class MirrorTemplateSchema(BaseModel):
    surface: SurfaceSchema
    bsdf: BSDFSchema | None = None


class MirrorSchema(BaseModel):
    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    template: str
    curvature: float | None = None
    conic: float | None = None
    aspheric: list[float] | None = None
    offset: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    stage: int = Field(ge=0, default=0)
    bsdf_scale: float | None = None
    bsdf_type: str | None = None
    id: str | None = None


# Lens schemas


class AsphericDiskLensSchema(BaseModel):
    type: Literal["aspheric_disk"] = "aspheric_disk"
    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    curvature: float
    conic: float
    n_inside: float = Field(gt=0)
    n_outside: float = Field(gt=0, default=1.0)
    aspheric: list[float] = Field(default_factory=list)
    offset: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    transmittance: float = Field(ge=0, le=1, default=1.0)
    stage: int = Field(ge=0, default=0)
    id: str | None = None


class PlanoSlabSchema(BaseModel):
    type: Literal["plano_slab"] = "plano_slab"
    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    thickness: float = Field(gt=0)
    n_inside: float = Field(gt=0)
    n_outside: float = Field(gt=0, default=1.0)
    transmittance: float = Field(ge=0, le=1, default=1.0)
    stage: int = Field(ge=0, default=0)
    id: str | None = None


LensSchema = Annotated[
    AsphericDiskLensSchema | PlanoSlabSchema,
    Field(discriminator="type"),
]


# Obstruction schemas


class CylinderObstructionSchema(BaseModel):
    type: Literal["cylinder"] = "cylinder"
    p1: Vec3
    p2: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class OpenCylinderObstructionSchema(BaseModel):
    type: Literal["open_cylinder"] = "open_cylinder"
    p1: Vec3
    p2: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class BoxObstructionSchema(BaseModel):
    type: Literal["box"] = "box"
    p1: Vec3
    p2: Vec3
    id: str | None = None


class SphereObstructionSchema(BaseModel):
    type: Literal["sphere"] = "sphere"
    center: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class OrientedBoxObstructionSchema(BaseModel):
    type: Literal["oriented_box"] = "oriented_box"
    center: Vec3
    half_extents: Vec3
    rotation: Vec3
    id: str | None = None


class TriangleObstructionSchema(BaseModel):
    type: Literal["triangle"] = "triangle"
    v0: Vec3
    v1: Vec3
    v2: Vec3
    id: str | None = None


ObstructionSchema = Annotated[
    CylinderObstructionSchema
    | OpenCylinderObstructionSchema
    | BoxObstructionSchema
    | SphereObstructionSchema
    | OrientedBoxObstructionSchema
    | TriangleObstructionSchema,
    Field(discriminator="type"),
]


# Sensor schemas


def _normalize_sensor_placement(data: Any) -> Any:
    """Fold singular ``position``/``orientation`` into the plural form.

    A sensor entry may write either the singular ``position``/``orientation``
    (N=1 shortcut) or the plural ``positions``/``orientations`` lists, but
    not both. After this validator runs the schema only stores the plural
    canonical form; downstream code never has to disambiguate.
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    for singular, plural in (("position", "positions"), ("orientation", "orientations")):
        has_s = data.get(singular) is not None
        has_p = data.get(plural) is not None
        if has_s == has_p:
            raise ValueError(
                f"Sensor entry must set exactly one of `{singular}` or `{plural}`."
            )
        if has_s:
            data[plural] = [data.pop(singular)]
        else:
            data.pop(singular, None)
    if len(data["positions"]) != len(data["orientations"]):
        raise ValueError(
            f"`positions` ({len(data['positions'])}) and "
            f"`orientations` ({len(data['orientations'])}) must have the same length."
        )
    return data


def _serialize_placement_singular(
    self: BaseModel, handler: SerializerFunctionWrapHandler
) -> dict[str, Any]:
    """Emit ``position``/``orientation`` for N=1 sensors, plural otherwise."""
    out = handler(self)
    for singular, plural in (("position", "positions"), ("orientation", "orientations")):
        values = out.get(plural)
        if isinstance(values, list) and len(values) == 1:
            out[singular] = values[0]
            del out[plural]
    return out


class SquareSensorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["square"] = "square"
    positions: list[Vec3] = Field(min_length=1)
    orientations: list[Vec3] = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bounds: Bounds4
    edge_width: float = Field(ge=0, default=0.0)
    id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_placement(cls, data: Any) -> Any:
        return _normalize_sensor_placement(data)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _serialize_placement_singular(self, handler)


class HexagonalSensorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hexagonal"] = "hexagonal"
    positions: list[Vec3] = Field(min_length=1)
    orientations: list[Vec3] = Field(min_length=1)
    centers_x: list[float] = Field(min_length=1)
    centers_y: list[float] = Field(min_length=1)
    edge_width: float = Field(ge=0, default=0.0)
    id: str | None = None

    @field_validator("centers_y")
    @classmethod
    def centers_same_length(cls, v, info):
        if "centers_x" in info.data and len(v) != len(info.data["centers_x"]):
            raise ValueError("centers_x and centers_y must have same length")
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_placement(cls, data: Any) -> Any:
        return _normalize_sensor_placement(data)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _serialize_placement_singular(self, handler)


SensorSchema = Annotated[
    SquareSensorSchema | HexagonalSensorSchema,
    Field(discriminator="type"),
]


# Camera schema


class CameraConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Top-level telescope schema


class TelescopeMetadataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "telescope"
    units: str = "m"
    camera_position: Vec3
    camera_rotation: Vec3


class TelescopeConfigSchema(BaseModel):
    """Top-level schema for a telescope-only YAML file.

    Describes the optical system (mirrors, lenses, obstructions) plus the
    camera frame (where rays should be delivered) in world coordinates.
    Sensor layout and detector response live in a separate camera file
    (see :class:`CameraFileSchema`).
    """

    model_config = ConfigDict(extra="forbid")

    telescope: TelescopeMetadataSchema
    mirror_templates: dict[str, MirrorTemplateSchema] = Field(default_factory=dict)
    mirrors: list[MirrorSchema] = Field(default_factory=list)
    lenses: list[LensSchema] = Field(default_factory=list)
    obstructions: list[ObstructionSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_template_references(self) -> TelescopeConfigSchema:
        """Validate that all mirror template references exist."""
        for i, mirror in enumerate(self.mirrors):
            mirror_id = mirror.id or f"mirror[{i}]"
            if mirror.template not in self.mirror_templates:
                available = ", ".join(self.mirror_templates.keys()) if self.mirror_templates else "(none defined)"
                raise ValueError(
                    f"Mirror '{mirror_id}' references undefined template "
                    f"'{mirror.template}'. Available templates: {available}"
                )
        return self

    @model_validator(mode="after")
    def validate_optical_stages(self) -> TelescopeConfigSchema:
        """Each optical stage may contain only one optical group.

        Mirrors and lenses are each grouped per stage by the adapter;
        having a mirror and a lens at the same stage, or elements of
        different aperture types at the same stage, would build two
        groups for that stage and trip the same check in
        ``Telescope.__init__`` with a far less actionable error.
        """
        def _aperture_sig(ap: ApertureSchema) -> tuple[str, int]:
            if isinstance(ap, PolygonApertureSchema):
                return ("polygon", len(ap.vertices))
            return ("circular", 0)

        # Collect mirror aperture signatures per stage.
        mirror_sigs: dict[int, set[tuple[str, int]]] = {}
        for i, mirror in enumerate(self.mirrors):
            sig = _aperture_sig(mirror.aperture)
            sigs = mirror_sigs.setdefault(mirror.stage, set())
            if sigs and sig not in sigs:
                mirror_id = mirror.id or f"mirror[{i}]"
                raise ValueError(
                    f"Mirror '{mirror_id}' at stage {mirror.stage} mixes "
                    f"aperture type {sig} with already-seen apertures "
                    f"{sorted(sigs)} at the same stage. Each optical "
                    f"stage must contain a single mirror aperture type "
                    f"(disk OR polygon-with-N-vertices)."
                )
            sigs.add(sig)

        # Collect lens aperture signatures per (type, stage). Multiple
        # lens types may not share a stage either; within one (type, stage)
        # bucket, all elements must share an aperture signature.
        lens_sigs: dict[tuple[str, int], set[tuple[str, int]]] = {}
        lens_stages: dict[int, str] = {}
        mirror_stages = set(mirror_sigs.keys())
        for i, lens in enumerate(self.lenses):
            lens_id = lens.id or f"lens[{i}]"

            if lens.stage in mirror_stages:
                raise ValueError(
                    f"Lens '{lens_id}' at stage {lens.stage} conflicts "
                    f"with a mirror at the same stage. Only one optical "
                    f"group per stage is allowed."
                )

            previous_type = lens_stages.setdefault(lens.stage, lens.type)
            if previous_type != lens.type:
                raise ValueError(
                    f"Lens '{lens_id}' of type '{lens.type}' at stage "
                    f"{lens.stage} conflicts with an earlier lens of type "
                    f"'{previous_type}' at the same stage. Only one optical "
                    f"group per stage is allowed."
                )

            sig = _aperture_sig(lens.aperture)
            sigs = lens_sigs.setdefault((lens.type, lens.stage), set())
            if sigs and sig not in sigs:
                raise ValueError(
                    f"Lens '{lens_id}' at stage {lens.stage} mixes "
                    f"aperture type {sig} with already-seen apertures "
                    f"{sorted(sigs)} at the same stage. Each optical "
                    f"stage must contain a single lens aperture type "
                    f"(disk OR polygon-with-N-vertices)."
                )
            sigs.add(sig)
        return self


# Top-level camera schema


class CameraFileSchema(BaseModel):
    """Top-level schema for a standalone camera YAML file.

    **Sensor positions are always in the camera-local frame.** There is no
    world-frame mode: loading interprets every ``sensors[*].position`` as
    an offset from the camera origin, and saving writes them the same way.
    Unknown top-level keys (e.g. a hopeful ``frame: world`` tag) raise
    :class:`ValidationError` thanks to ``extra="forbid"``.

    The telescope and the camera frame live in the telescope YAML
    (``TelescopeConfigSchema.telescope.camera_position / camera_rotation``);
    the camera file knows nothing about the world.
    """

    model_config = ConfigDict(extra="forbid")

    camera: CameraConfigSchema = Field(default_factory=CameraConfigSchema)
    sensors: list[SensorSchema] = Field(default_factory=list)
