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


class ZernikeSchema(BaseModel):
    """Per-element Zernike figure error summed onto the aspheric surface.
    ``coeffs`` are RMS-normalized Noll coefficients in metres, indexed from
    ``Z1`` (``coeffs[0]`` = piston). At most 11 terms (Z1..Z11) are supported.
    ``r_norm`` is the normalization radius, with ``rho = 1`` at this radius.
    See :class:`~iactrace.core.surfaces.ZernikeSurfaceGroup`. Presence of this
    block makes the element's surface an asphere + Zernike
    :class:`~iactrace.core.surfaces.SumSurfaceGroup`.
    """

    coeffs: list[float] = Field(min_length=1, max_length=11)
    r_norm: float = Field(gt=0)


class GaussianBSDFSchema(BaseModel):
    """Single-Gaussian surface roughness.

    See :class:`~iactrace.core.bsdf.GaussianBSDF`.
    """

    type: Literal["gaussian"] = "gaussian"
    scale: float = Field(ge=0, default=0.0)


class DoubleGaussianBSDFSchema(BaseModel):
    """Two-component (narrow + wide) Gaussian roughness mixture.

    See :class:`~iactrace.core.bsdf.DoubleGaussianBSDF`.
    """

    type: Literal["double_gaussian"] = "double_gaussian"
    scale_narrow: float = Field(ge=0, default=0.0)
    scale_wide: float = Field(ge=0, default=0.0)
    mix_weight: float = Field(ge=0, le=1, default=0.0)


BSDFSchema = Annotated[
    GaussianBSDFSchema | DoubleGaussianBSDFSchema,
    Field(discriminator="type"),
]


class TabulatedCurveSchema(BaseModel):
    """Inline tabulated angle-dependent coating curve.

    ``angles_deg`` are sample angles in degrees in ``[0, 90]``;
    ``values`` are the corresponding coefficients in ``[0, 1]``. The
    two lists must have the same length. See
    :class:`~iactrace.core.coatings.TabulatedCoating`.
    """

    type: Literal["table"] = "table"
    angles_deg: list[float] = Field(min_length=2)
    values: list[float] = Field(min_length=2)

    @field_validator("angles_deg")
    @classmethod
    def _angles_in_range(cls, v):
        for a in v:
            if a < 0.0 or a > 90.0:
                raise ValueError(f"angles_deg must lie in [0, 90]; got {a}")
        return v

    @field_validator("values")
    @classmethod
    def _values_in_range(cls, v):
        for x in v:
            if x < 0.0 or x > 1.0:
                raise ValueError(f"values must lie in [0, 1]; got {x}")
        return v

    @model_validator(mode="after")
    def _same_length(self) -> TabulatedCurveSchema:
        if len(self.angles_deg) != len(self.values):
            raise ValueError(
                f"angles_deg ({len(self.angles_deg)}) and values "
                f"({len(self.values)}) must have the same length"
            )
        return self


class MirrorTemplateSchema(BaseModel):
    """Deduplicated surface geometry shared by mirrors. Per-element coating
    (``reflectivity``) and ``bsdf`` live on the mirror, not the template."""

    surface: SurfaceSchema
    bsdf: BSDFSchema | None = None
    reflectivity: float | None = None
    coating: TabulatedCurveSchema | None = None


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
    bsdf: BSDFSchema | None = None
    reflectivity: float | None = None
    zernike: ZernikeSchema | None = None
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
    coating: TabulatedCurveSchema | None = None
    zernike: ZernikeSchema | None = None
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
    coating: TabulatedCurveSchema | None = None
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


# Detection-chain schemas (per sensor group)


class WinstonConeSchema(BaseModel):
    """Serialized :class:`~iactrace.camera.winston_cone.WinstonCone`.

    ``entrance_apothem`` is the physical mouth at ``z = length`` (the truncated
    mouth when ``length`` is given, the full CPC mouth when it is omitted). The
    wall tilt (hence the parabola) is derived from ``(exit_apothem,
    entrance_apothem, length)``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["winston"] = "winston"
    n_sides: int = Field(gt=2)
    entrance_apothem: float = Field(gt=0)
    exit_apothem: float = Field(gt=0)
    length: float | None = Field(default=None, gt=0)
    reflectivity: float = Field(ge=0, le=1, default=0.9)
    max_bounces: int = Field(ge=0, default=10)
    orientation_deg: float = 0.0


class OkumuraConeSchema(BaseModel):
    """Serialized :class:`~iactrace.camera.okumura_cone.OkumuraCone`.

    The walls follow a quadratic or cubic Bezier meridian instead of Winston's
    paraboloid. ``control_points`` are the interior Bezier points in the paper's
    normalized box (exit rim ``(0, 0)``, mouth ``(1, 1)`` implied) -- one point
    for a quadratic curve, two for a cubic one. ``length`` defaults to the
    equivalent full Winston-cone depth when omitted.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["okumura"] = "okumura"
    n_sides: int = Field(gt=2)
    entrance_apothem: float = Field(gt=0)
    exit_apothem: float = Field(gt=0)
    control_points: list[Vec2] = Field(min_length=1)
    length: float | None = Field(default=None, gt=0)
    reflectivity: float = Field(ge=0, le=1, default=0.9)
    max_bounces: int = Field(ge=0, default=10)
    orientation_deg: float = 0.0


class ConstantQESchema(BaseModel):
    """Serialized :class:`~iactrace.camera.photosensor.ConstantQE`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["constant"] = "constant"
    qe: float = Field(ge=0, le=1, default=1.0)


# Discriminated-union slots for the detection chain. ``ConcentratorSchema`` now
# carries two concrete members (Winston + Okumura), keyed on the ``type`` literal;
# ``io/adapters.py`` has the matching build/dump arms. The photosensor slot stays
# a single member (``ConstantQE``) until a second concrete type joins it.
ConcentratorSchema = Annotated[
    WinstonConeSchema | OkumuraConeSchema,
    Field(discriminator="type"),
]
PhotoSensorSchema = ConstantQESchema


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
            raise ValueError(f"Sensor entry must set exactly one of `{singular}` or `{plural}`.")
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
    """Emit ``position``/``orientation`` for N=1 sensors, plural otherwise.

    Also drops a zero ``gap`` so geometry-only sensor groups (no concentrator,
    no gap, perfect QE) round-trip without spurious detection-chain keys.
    """
    out = handler(self)
    for singular, plural in (("position", "positions"), ("orientation", "orientations")):
        values = out.get(plural)
        if isinstance(values, list) and len(values) == 1:
            out[singular] = values[0]
            del out[plural]
    if out.get("gap") == 0.0:
        out.pop("gap", None)
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
    concentrator: ConcentratorSchema | None = None
    gap: float = Field(ge=0, default=0.0)
    photosensor: PhotoSensorSchema | None = None
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
    concentrator: ConcentratorSchema | None = None
    gap: float = Field(ge=0, default=0.0)
    photosensor: PhotoSensorSchema | None = None
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
                available = (
                    ", ".join(self.mirror_templates.keys())
                    if self.mirror_templates
                    else "(none defined)"
                )
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

    A camera file is a list of ``sensors``; each sensor group carries its own
    detector geometry **and** detection chain (``concentrator`` / ``gap`` /
    ``photosensor``), so different groups can run different chains.

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

    sensors: list[SensorSchema] = Field(default_factory=list)
