"""The per-pixel detection chain: concentrators and photodetectors."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from ...camera.detector import PMT, ConstantQE, TabulatedQE
from ...camera.optics import OkumuraCone, WinstonCone
from ...camera.optics.winston import cpc_full_length
from ...core.responses import ResponseCurve
from ..schemas import (
    ConcentratorSchema,
    ConstantQESchema,
    OkumuraConeSchema,
    PhotoDetectorSchema,
    PMTSchema,
    TabulatedCurveSchema,
    TabulatedQESchema,
    WinstonConeSchema,
)
from .curves import (
    _build_curve_for_bucket,
    _build_index_for_bucket,
    _curve_to_schema,
    _index_to_schema,
)
from .surfaces import _single_element_surface, _surface_components, _surface_to_spec

if TYPE_CHECKING:
    from ...camera.detection_chain import DetectionChain
    from ...camera.detector import PhotoDetector
    from ...camera.optics import Concentrator


class _ConcentratorSpec(NamedTuple):
    """Bidirectional spec for one concentrator type, mirroring _ObsSpec.

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


def _wall_curve_to_schema(concentrator: Concentrator) -> TabulatedCurveSchema | None:
    """Serialise a concentrator's optional wall reflectivity_curve.

    Reuses _curve_to_schema (the cone holds a single-element
    ResponseCurve), so a ConstantResponse or no
    coating round-trips as None and a wavelength curve emits the same
    {type: table, ...} form used for mirror / lens coatings.
    """
    return _curve_to_schema(getattr(concentrator, "reflectivity_curve", None))


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
        reflectivity_curve=_wall_curve_to_schema(concentrator),
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
        reflectivity_curve=_build_curve_for_bucket([schema.reflectivity_curve], 1),
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
        reflectivity_curve=_wall_curve_to_schema(concentrator),
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
        reflectivity_curve=_build_curve_for_bucket([schema.reflectivity_curve], 1),
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
    """Serialize a concentrator (None -> None); see _CONCENTRATOR_SPECS.

    Raises for any Concentrator
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
    """Rebuild a concentrator from its schema (None -> no concentrator)."""
    if schema is None:
        return None
    spec = _CONCENTRATOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:
        raise ValueError(f"unknown concentrator schema: {type(schema).__name__}")
    return spec.from_schema(schema)


class _PhotoDetectorSpec(NamedTuple):
    """Bidirectional spec for one photodetector type; see _ConcentratorSpec."""

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


def _qe_curve_to_schema(qe_curve: ResponseCurve, owner: str) -> TabulatedCurveSchema:
    """Serialise a photodetector's qe_curve, or explain why it cannot be."""
    schema = _curve_to_schema(qe_curve)
    if schema is None:
        raise ValueError(
            f"Cannot serialise the qe_curve of {owner} to YAML: only a "
            "TabulatedResponse has a written form."
        )
    return schema


def _tabulated_qe_to_schema(photodetector: TabulatedQE) -> TabulatedQESchema:
    return TabulatedQESchema(
        qe=float(photodetector.qe),
        qe_curve=_qe_curve_to_schema(photodetector.qe_curve, "TabulatedQE"),
    )


def _tabulated_qe_from_schema(schema: TabulatedQESchema) -> TabulatedQE:
    curve = _build_curve_for_bucket([schema.qe_curve], 1)
    assert curve is not None  # the schema requires a curve
    return TabulatedQE(qe_curve=curve, qe=schema.qe)


def _pmt_to_schema(photodetector: PMT) -> PMTSchema:
    # The photocathode figure round-trips through the exact same
    # (aspheric, zernike) decomposition mirrors and lenses use.
    asph, zern = _surface_components(photodetector.shape)
    qe_curve = (
        _qe_curve_to_schema(photodetector.qe_curve, "PMT")
        if photodetector.qe_curve is not None
        else None
    )
    window_index = (
        _index_to_schema(photodetector.window_index, 0)
        if photodetector.window_index is not None
        else None
    )
    return PMTSchema(
        qe=float(photodetector.qe),
        qe_curve=qe_curve,
        window_index=window_index,
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
        qe_curve=(
            _build_curve_for_bucket([schema.qe_curve], 1) if schema.qe_curve is not None else None
        ),
        window_index=(
            _build_index_for_bucket([schema.window_index], 1)
            if schema.window_index is not None
            else None
        ),
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
    _PhotoDetectorSpec(
        "tabulated",
        TabulatedQESchema,
        TabulatedQE,
        _tabulated_qe_to_schema,
        _tabulated_qe_from_schema,
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
    """Serialize a photodetector; see _PHOTODETECTOR_SPECS.

    Raises for any PhotoDetector
    subclass without a registered spec, rather than silently falling back to
    a perfect ConstantQE(1.0) -- that fallback would be a large, silent
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
    """Rebuild a photodetector from its schema (None -> ConstantQE(1.0))."""
    if schema is None:
        return ConstantQE(1.0)
    spec = _PHOTODETECTOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:
        raise ValueError(f"unknown photodetector schema: {type(schema).__name__}")
    return spec.from_schema(schema)


def _chain_to_schema_fields(
    chain: DetectionChain,
) -> tuple[ConcentratorSchema | None, float, PhotoDetectorSchema | None]:
    """Project a detection chain to its (concentrator, gap, photodetector) schema.

    Photodetectors (ConstantQE /
    PMT) and concentrators
    (WinstonCone /
    OkumuraCone) round-trip exactly;
    any other subclass raises (see _photodetector_to_schema /
    _concentrator_to_schema) rather than silently degrading the physics.
    The trivial perfect-QE photodetector is emitted as None so a
    geometry-only sensor group serializes without a redundant
    photodetector: block.
    """
    concentrator = _concentrator_to_schema(chain.concentrator)
    photodetector: PhotoDetectorSchema | None
    if isinstance(chain.photodetector, ConstantQE) and float(chain.photodetector.qe) == 1.0:
        photodetector = None
    else:
        photodetector = _photodetector_to_schema(chain.photodetector)
    return concentrator, float(chain.gap), photodetector
