"""Sensor groups and their pixel layout."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ...camera.sensor_group import HexagonalSensorGroup, SquareSensorGroup
from ..schemas import HexagonalSensorSchema, SquareSensorSchema
from ._common import _to_float_list
from .detectors import (
    _chain_to_schema_fields,
    _concentrator_from_schema,
    _photodetector_from_schema,
)

if TYPE_CHECKING:
    from ...camera.sensor_group import SensorGroup

SensorSchemaType = SquareSensorSchema | HexagonalSensorSchema


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
    """Convert a validated sensor schema to a SensorGroup domain object; see _SENSOR_SPECS.

    The schema position / orientation are interpreted as
    camera-local coordinates.
    """
    positions = [list(p) for p in schema.positions]
    rotations = [list(r) for r in schema.orientations]
    concentrator = _concentrator_from_schema(schema.concentrator)
    photodetector = _photodetector_from_schema(schema.photodetector)
    spec = _SENSOR_SPEC_BY_TYPE.get(schema.type)
    if spec is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise ValueError(f"unknown sensor schema type: {schema.type!r}")
    return spec.build(schema, positions, rotations, concentrator, photodetector, schema.gap)


def sensors_to_schemas(
    sensors: list[SensorGroup],
) -> list[SensorSchemaType]:
    """Extract sensor schemas from a SensorGroup list; see _SENSOR_SPECS.

    One YAML entry per SensorGroup: groups carrying multiple
    sensors are written with plural positions/orientations lists,
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
    """Bidirectional spec for one sensor-group type; see _ConcentratorSpec."""

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
