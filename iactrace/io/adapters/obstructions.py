"""Shadowing primitives."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ...core.obstructions import (
    BoxGroup,
    CylinderGroup,
    ObstructionGroup,
    OpenCylinderGroup,
    OrientedBoxGroup,
    SphereGroup,
    TriangleGroup,
)
from ...core.transforms import euler_to_matrix
from ..schemas import (
    BoxObstructionSchema,
    CylinderObstructionSchema,
    OpenCylinderObstructionSchema,
    OrientedBoxObstructionSchema,
    SphereObstructionSchema,
    TriangleObstructionSchema,
)
from ._common import _to_float_list

ObstructionSchemaType = (
    CylinderObstructionSchema
    | OpenCylinderObstructionSchema
    | BoxObstructionSchema
    | SphereObstructionSchema
    | OrientedBoxObstructionSchema
    | TriangleObstructionSchema
)


class _ObsField(NamedTuple):
    """One field of an obstruction, mapping a schema attr to a group attr.

    kind selects how the value is projected in each direction:
    vec3/scalar copy through (arrays batch on load, elements read
    back on save); euler_matrix converts Euler degrees <-> a 3x3 matrix.
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
        "cylinder",
        CylinderObstructionSchema,
        CylinderGroup,
        (
            _ObsField("p1", "p1", "vec3"),
            _ObsField("p2", "p2", "vec3"),
            _ObsField("r", "r", "scalar"),
        ),
    ),
    _ObsSpec(
        "open_cylinder",
        OpenCylinderObstructionSchema,
        OpenCylinderGroup,
        (
            _ObsField("p1", "p1", "vec3"),
            _ObsField("p2", "p2", "vec3"),
            _ObsField("r", "r", "scalar"),
        ),
    ),
    _ObsSpec(
        "box",
        BoxObstructionSchema,
        BoxGroup,
        (_ObsField("p1", "p1", "vec3"), _ObsField("p2", "p2", "vec3")),
    ),
    _ObsSpec(
        "sphere",
        SphereObstructionSchema,
        SphereGroup,
        (_ObsField("center", "centers", "vec3"), _ObsField("r", "radii", "scalar")),
    ),
    _ObsSpec(
        "oriented_box",
        OrientedBoxObstructionSchema,
        OrientedBoxGroup,
        (
            _ObsField("center", "centers", "vec3"),
            _ObsField("half_extents", "half_extents", "vec3"),
            _ObsField("rotation", "rotations", "euler_matrix"),
        ),
    ),
    _ObsSpec(
        "triangle",
        TriangleObstructionSchema,
        TriangleGroup,
        (
            _ObsField("v0", "v0", "vec3"),
            _ObsField("v1", "v1", "vec3"),
            _ObsField("v2", "v2", "vec3"),
        ),
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
    _OBSTRUCTION_SPECS declaration order, independent of input order.
    """
    by_type: dict[str, list] = defaultdict(list)
    for obs in obstructions:
        by_type[obs.type].append(obs)
    return [
        _build_obstruction_group(spec, by_type[spec.type_name])
        for spec in _OBSTRUCTION_SPECS
        if by_type.get(spec.type_name)
    ]


_SPEC_BY_GROUP: dict[type, _ObsSpec] = {spec.group: spec for spec in _OBSTRUCTION_SPECS}


def matrix_to_euler(rotation_matrix) -> list[float]:
    """Decompose a 3x3 rotation into XYZ Euler angles in degrees.

    The inverse of euler_to_matrix, i.e. it
    decomposes R = Rz(rz) @ Ry(ry) @ Rx(rx). Host-side only: it branches on
    the matrix values and returns Python floats, so it cannot be traced.
    """
    sy = jnp.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy > 1e-6:
        rx = jnp.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        ry = jnp.arctan2(-rotation_matrix[2, 0], sy)
        rz = jnp.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        rx = jnp.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        ry = jnp.arctan2(-rotation_matrix[2, 0], sy)
        rz = jnp.array(0.0)
    return [float(jnp.degrees(rx)), float(jnp.degrees(ry)), float(jnp.degrees(rz))]


def _extract_obstruction(spec: _ObsSpec, group: ObstructionGroup, i: int, counter: int):
    """Project element i of an obstruction group back to its schema."""
    kwargs: dict[str, object] = {"id": f"obs_{counter}"}
    for f in spec.fields:
        col = getattr(group, f.group_attr)
        if f.kind == "scalar":
            kwargs[f.schema_attr] = float(col[i])
        elif f.kind == "euler_matrix":
            kwargs[f.schema_attr] = matrix_to_euler(np.asarray(col[i]))
        else:  # vec3
            kwargs[f.schema_attr] = _to_float_list(col[i])
    return spec.schema(**kwargs)


def obstructions_to_schemas(
    groups: list[ObstructionGroup] | None,
) -> list[ObstructionSchemaType]:
    """Extract obstruction schemas from an ObstructionGroup list.

    One schema per primitive, id-numbered globally in traversal order.
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
