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


class ConfigModel(BaseModel):
    """Base for every config schema."""

    model_config = ConfigDict(extra="forbid")


# 1. Shared pieces: constrained vectors and apertures

Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Bounds4 = Annotated[list[float], Field(min_length=4, max_length=4)]


# Apertures


class CircularApertureSchema(ConfigModel):
    type: Literal["circular"] = "circular"
    radius: float = Field(gt=0)
    inner_radius: float = Field(ge=0, default=0.0)


class PolygonApertureSchema(ConfigModel):
    type: Literal["polygon"] = "polygon"
    vertices: list[Vec2] = Field(min_length=3)


ApertureSchema = Annotated[
    CircularApertureSchema | PolygonApertureSchema,
    Field(discriminator="type"),
]


# 2. Surface shapes


class AsphericSurfaceSchema(ConfigModel):
    """Even-aspheric conic surface"""

    type: Literal["aspheric"] = "aspheric"
    curvature: float
    conic: float = 0.0
    aspheric: list[float] = Field(default_factory=list)


class ZernikeSurfaceSchema(ConfigModel):
    """Zernike figure surface (RMS-normalized Noll coefficients, in metres).

    coeffs are indexed from Z1 (coeffs[0] = piston); at most 11
    terms (Z1..Z11) are supported. r_norm is the normalization radius, with
    rho = 1 at this radius.
    """

    type: Literal["zernike"] = "zernike"
    coeffs: list[float] = Field(min_length=1, max_length=11)
    r_norm: float = Field(gt=0)


# A single surface shape.
SurfaceSchema = Annotated[
    AsphericSurfaceSchema | ZernikeSurfaceSchema,
    Field(discriminator="type"),
]

SurfaceSpec = SurfaceSchema | list[SurfaceSchema]


# 3. Surface roughness


class GaussianBSDFSchema(ConfigModel):
    """Single-Gaussian surface roughness.

    See GaussianBSDF.
    """

    type: Literal["gaussian"] = "gaussian"
    scale: float = Field(ge=0, default=0.0)


class DoubleGaussianBSDFSchema(ConfigModel):
    """Two-component (narrow + wide) Gaussian roughness mixture.

    See DoubleGaussianBSDF.
    """

    type: Literal["double_gaussian"] = "double_gaussian"
    scale_narrow: float = Field(ge=0, default=0.0)
    scale_wide: float = Field(ge=0, default=0.0)
    mix_weight: float = Field(ge=0, le=1, default=0.0)


BSDFSchema = Annotated[
    GaussianBSDFSchema | DoubleGaussianBSDFSchema,
    Field(discriminator="type"),
]


# 4. Response curves and refractive indices


class TabulatedCurveSchema(ConfigModel):
    """Inline tabulated response curve: angle-only or an angle-wavelength grid.

    angles_deg are sample angles in degrees in [0, 90] and all
    coefficients are in [0, 1].

    - Angle-only (wavelengths_nm omitted): values is a 1-D list
      aligned with angles_deg -- R(theta).
    - Angle-wavelength grid (wavelengths_nm given): values is a 2-D
      list with values[i][j] = R(angles_deg[i], wavelengths_nm[j]).

    See TabulatedResponse.
    """

    type: Literal["table"] = "table"
    angles_deg: list[float] = Field(min_length=1)
    values: list[float] | list[list[float]] = Field(min_length=1)
    wavelengths_nm: list[float] | None = None

    @field_validator("angles_deg")
    @classmethod
    def _angles_in_range(cls, v):
        for a in v:
            if a < 0.0 or a > 90.0:
                raise ValueError(f"angles_deg must lie in [0, 90]; got {a}")
        return v

    @field_validator("wavelengths_nm")
    @classmethod
    def _wavelengths_positive(cls, v):
        if v is not None:
            if len(v) < 1:
                raise ValueError("wavelengths_nm must be non-empty when given")
            for w in v:
                if w <= 0.0:
                    raise ValueError(f"wavelengths_nm must be > 0; got {w}")
        return v

    @model_validator(mode="after")
    def _check_shape(self) -> TabulatedCurveSchema:
        if self.wavelengths_nm is None:
            # Angle-only: a flat list of coefficients, one per angle.
            if len(self.values) != len(self.angles_deg):
                raise ValueError(
                    f"angles_deg ({len(self.angles_deg)}) and values "
                    f"({len(self.values)}) must have the same length"
                )
            for x in self.values:
                if isinstance(x, list):
                    raise ValueError(
                        "values must be a 1-D list when wavelengths_nm is omitted; "
                        "give wavelengths_nm for an (angle, wavelength) grid"
                    )
                if not 0.0 <= x <= 1.0:
                    raise ValueError(f"values must lie in [0, 1]; got {x}")
        else:
            # Grid: one row per angle, one column per wavelength.
            nwl = len(self.wavelengths_nm)
            if len(self.values) != len(self.angles_deg):
                raise ValueError(
                    f"values must have one row per angle: got {len(self.values)} "
                    f"rows for {len(self.angles_deg)} angles"
                )
            for row in self.values:
                if not isinstance(row, list) or len(row) != nwl:
                    raise ValueError(
                        f"each values row must be a list of length len(wavelengths_nm)={nwl}"
                    )
                for x in row:
                    if not 0.0 <= x <= 1.0:
                        raise ValueError(f"values must lie in [0, 1]; got {x}")
        return self


# Discriminated union of response curves.
ResponseCurveSchema = Annotated[
    TabulatedCurveSchema,
    Field(discriminator="type"),
]


class SellmeierIndexSchema(ConfigModel):
    """Sellmeier n(lambda) coefficients for a dispersive glass.

    n(lambda)2 = 1 + sum_j b_j lambda2 / (lambda**2 - c_j). The c
    coefficients carry units of wavelength squared and must match the bundle's
    wavelength unit (nanometres by convention). See
    SellmeierIndex.
    """

    type: Literal["sellmeier"] = "sellmeier"
    b: list[float] = Field(min_length=1)
    c: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _same_length(self) -> SellmeierIndexSchema:
        if len(self.b) != len(self.c):
            raise ValueError(f"b ({len(self.b)}) and c ({len(self.c)}) must have equal length")
        return self


class TabulatedIndexSchema(ConfigModel):
    """Tabulated n(lambda) samples, linearly interpolated in wavelength.

    See TabulatedIndex.
    """

    type: Literal["index_table"] = "index_table"
    wavelengths_nm: list[float] = Field(min_length=2)
    n: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def _same_length(self) -> TabulatedIndexSchema:
        if len(self.wavelengths_nm) != len(self.n):
            raise ValueError(
                f"wavelengths_nm ({len(self.wavelengths_nm)}) and n ({len(self.n)}) "
                "must have the same length"
            )
        return self


# A dispersive refractive-index model.
RefractiveIndexSchema = Annotated[
    SellmeierIndexSchema | TabulatedIndexSchema,
    Field(discriminator="type"),
]

IndexSchema = Annotated[float, Field(gt=0)] | RefractiveIndexSchema


# 5. Mirrors: a template of shared defaults, and the facets themselves


class MirrorTemplateSchema(ConfigModel):
    """Optional shared defaults a mirror may reference via template.

    Every field here can also be set directly on the mirror; see
    MirrorSchema for the override rule. A template with no
    surface is valid (e.g. one that only shares a reflectivity_curve), since a
    mirror without an aspheric base of its own defaults to flat.
    """

    surface: SurfaceSpec | None = None
    bsdf: BSDFSchema | None = None
    reflectivity: float | None = None
    reflectivity_curve: ResponseCurveSchema | None = None


class MirrorSchema(ConfigModel):
    """A mirror facet.

    Fully self-contained by default: curvature / conic / aspheric
    / zernike / bsdf / reflectivity / reflectivity_curve can all be set
    directly here, with no template required. template (optional)
    names a MirrorTemplateSchema entry supplying defaults for
    whichever of those fields the mirror itself leaves unset -- a mirror's
    own value always wins when both are defined; a field left unset on both
    falls back to its ordinary default (flat / unmodified surface, perfect
    specular reflection, bare Fresnel-free reflectivity of 1.0).
    """

    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    template: str | None = None
    curvature: float | None = None
    conic: float | None = None
    aspheric: list[float] | None = None
    zernike: ZernikeSurfaceSchema | None = None
    offset: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    stage: int = Field(ge=0, default=0)
    bsdf: BSDFSchema | None = None
    reflectivity: float | None = None
    reflectivity_curve: ResponseCurveSchema | None = None
    id: str | None = None


# 6. Lenses


class AsphericDiskLensSchema(ConfigModel):
    type: Literal["aspheric_disk"] = "aspheric_disk"
    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    surface: SurfaceSpec
    index: IndexSchema
    offset: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    transmittance: float = Field(ge=0, le=1, default=1.0)
    transmittance_curve: ResponseCurveSchema | None = None
    stage: int = Field(ge=0, default=0)
    id: str | None = None


class PlanoSlabSchema(ConfigModel):
    type: Literal["plano_slab"] = "plano_slab"
    position: Vec3
    orientation: Vec3
    aperture: ApertureSchema
    thickness: float = Field(gt=0)
    index: IndexSchema
    transmittance: float = Field(ge=0, le=1, default=1.0)
    transmittance_curve: ResponseCurveSchema | None = None
    stage: int = Field(ge=0, default=0)
    id: str | None = None


LensSchema = Annotated[
    AsphericDiskLensSchema | PlanoSlabSchema,
    Field(discriminator="type"),
]


# 7. Obstructions


class CylinderObstructionSchema(ConfigModel):
    type: Literal["cylinder"] = "cylinder"
    p1: Vec3
    p2: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class OpenCylinderObstructionSchema(ConfigModel):
    type: Literal["open_cylinder"] = "open_cylinder"
    p1: Vec3
    p2: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class BoxObstructionSchema(ConfigModel):
    type: Literal["box"] = "box"
    p1: Vec3
    p2: Vec3
    id: str | None = None


class SphereObstructionSchema(ConfigModel):
    type: Literal["sphere"] = "sphere"
    center: Vec3
    r: float = Field(gt=0)
    id: str | None = None


class OrientedBoxObstructionSchema(ConfigModel):
    type: Literal["oriented_box"] = "oriented_box"
    center: Vec3
    half_extents: Vec3
    rotation: Vec3
    id: str | None = None


class TriangleObstructionSchema(ConfigModel):
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


# 8. The detection chain, per sensor group


class WinstonConeSchema(ConfigModel):
    """Serialized WinstonCone.

    entrance_apothem is the physical mouth at z = length (the truncated
    mouth when length is given, the full CPC mouth when it is omitted). The
    wall tilt (hence the parabola) is derived from (exit_apothem,
    entrance_apothem, length).
    """

    type: Literal["winston"] = "winston"
    n_sides: int = Field(gt=2)
    entrance_apothem: float = Field(gt=0)
    exit_apothem: float = Field(gt=0)
    length: float | None = Field(default=None, gt=0)
    reflectivity: float = Field(ge=0, le=1, default=0.9)
    reflectivity_curve: ResponseCurveSchema | None = None
    max_bounces: int = Field(ge=0, default=10)
    orientation_deg: float = 0.0


class OkumuraConeSchema(ConfigModel):
    """Serialized OkumuraCone.

    The walls follow a quadratic or cubic Bezier meridian instead of Winston's
    paraboloid. control_points are the interior Bezier points in the paper's
    normalized box (exit rim (0, 0), mouth (1, 1) implied) -- one point
    for a quadratic curve, two for a cubic one. length defaults to the
    equivalent full Winston-cone depth when omitted.
    """

    type: Literal["okumura"] = "okumura"
    n_sides: int = Field(gt=2)
    entrance_apothem: float = Field(gt=0)
    exit_apothem: float = Field(gt=0)
    control_points: list[Vec2] = Field(min_length=1)
    length: float | None = Field(default=None, gt=0)
    reflectivity: float = Field(ge=0, le=1, default=0.9)
    reflectivity_curve: ResponseCurveSchema | None = None
    max_bounces: int = Field(ge=0, default=10)
    orientation_deg: float = 0.0


class ConstantQESchema(ConfigModel):
    """Serialized ConstantQE."""

    type: Literal["constant"] = "constant"
    qe: float = Field(ge=0, le=1, default=1.0)


class TabulatedQESchema(ConfigModel):
    """Serialized TabulatedQE.

    The detector-side qe / qe_curve pair: qe is the bulk scalar and
    qe_curve the response curve multiplying it, in the same
    {type: table, ...} form a mirror's reflectivity_curve takes. The
    legacy inline form (wavelengths_nm plus a list of qe values) is
    still accepted and folded into an angle-flat curve.
    """

    type: Literal["tabulated"] = "tabulated"
    qe: float = Field(ge=0, le=1, default=1.0)
    qe_curve: ResponseCurveSchema


class PMTSchema(ConfigModel):
    """Serialized PMT.

    A photomultiplier response: the qe / qe_curve photocathode pair
    plus an optional entrance index window (Fresnel angular response) --
    a plain number for a non-dispersive window, or a model for a dispersive
    one, the same field either way.
    """

    type: Literal["pmt"] = "pmt"
    qe: float = Field(ge=0, le=1, default=1.0)
    qe_curve: ResponseCurveSchema | None = None
    window_index: IndexSchema | None = None
    face_radius: float = Field(gt=0)
    surface: SurfaceSpec = Field(
        default_factory=lambda: AsphericSurfaceSchema(curvature=0.0, conic=0.0)
    )
    vertex_z: float = 0.0
    length: float | None = Field(default=None, ge=0)
    n_facets: int = Field(ge=3, default=48)


# Discriminated-union slots for the detection chain
ConcentratorSchema = Annotated[
    WinstonConeSchema | OkumuraConeSchema,
    Field(discriminator="type"),
]
PhotoDetectorSchema = Annotated[
    ConstantQESchema | TabulatedQESchema | PMTSchema,
    Field(discriminator="type"),
]


# 9. Sensors


def _normalize_sensor_placement(data: Any) -> Any:
    """Fold singular position/orientation into the plural form.

    A sensor entry may write either the singular position/orientation
    (N=1 shortcut) or the plural positions/orientations lists, but
    not both.
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
    """Emit position/orientation for N=1 sensors, plural otherwise.

    Also drops a zero gap so geometry-only sensor groups (no concentrator,
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


class SquareSensorSchema(ConfigModel):
    type: Literal["square"] = "square"
    positions: list[Vec3] = Field(min_length=1)
    orientations: list[Vec3] = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bounds: Bounds4
    edge_width: float = Field(ge=0, default=0.0)
    concentrator: ConcentratorSchema | None = None
    gap: float = Field(ge=0, default=0.0)
    photodetector: PhotoDetectorSchema | None = None
    id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_placement(cls, data: Any) -> Any:
        return _normalize_sensor_placement(data)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _serialize_placement_singular(self, handler)


class HexagonalSensorSchema(ConfigModel):
    type: Literal["hexagonal"] = "hexagonal"
    positions: list[Vec3] = Field(min_length=1)
    orientations: list[Vec3] = Field(min_length=1)
    centers_x: list[float] = Field(min_length=1)
    centers_y: list[float] = Field(min_length=1)
    edge_width: float = Field(ge=0, default=0.0)
    concentrator: ConcentratorSchema | None = None
    gap: float = Field(ge=0, default=0.0)
    photodetector: PhotoDetectorSchema | None = None
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


# 10. The top-level documents: a telescope file, and a camera file


class TelescopeMetadataSchema(ConfigModel):
    name: str = "telescope"
    units: str = "m"
    camera_position: Vec3
    camera_rotation: Vec3


class TelescopeConfigSchema(ConfigModel):
    """Top-level schema for a telescope-only YAML file.

    Describes the optical system (mirrors, lenses, obstructions) plus the
    camera frame (where rays should be delivered) in world coordinates.
    Sensor layout and detector response live in a separate camera file
    (see CameraFileSchema).
    """

    telescope: TelescopeMetadataSchema
    mirror_templates: dict[str, MirrorTemplateSchema] = Field(default_factory=dict)
    mirrors: list[MirrorSchema] = Field(default_factory=list)
    lenses: list[LensSchema] = Field(default_factory=list)
    obstructions: list[ObstructionSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_template_references(self) -> TelescopeConfigSchema:
        """Validate that all mirror template references exist.

        A mirror with no template (fully self-contained) has nothing to
        check here.
        """
        for i, mirror in enumerate(self.mirrors):
            if mirror.template is None:
                continue
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
        Telescope.__init__ with a far less actionable error.
        """

        def _aperture_sig(ap: ApertureSchema) -> tuple[str, int]:
            if ap.type == "polygon":
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


class CameraFileSchema(ConfigModel):
    """Top-level schema for a standalone camera YAML file.

    A camera file is a list of sensors; each sensor group carries its own
    detector geometry and detection chain (concentrator / gap /
    photosensor), so different groups can run different chains.

    Sensor positions are always in the camera-local frame. There is no
    world-frame mode: loading interprets every sensors[*].position as
    an offset from the camera origin, and saving writes them the same way.
    Unknown top-level keys (e.g. a hopeful frame: world tag) raise
    ValidationError thanks to extra="forbid".

    The telescope and the camera frame live in the telescope YAML
    (TelescopeConfigSchema.telescope.camera_position / camera_rotation);
    the camera file knows nothing about the world.
    """

    sensors: list[SensorSchema] = Field(default_factory=list)
