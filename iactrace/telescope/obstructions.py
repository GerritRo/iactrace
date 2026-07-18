from __future__ import annotations

from collections.abc import Sequence

# Re-export the low-level primitives for convenience
from ..core.obstructions import (
    BoxGroup,
    CylinderGroup,
    ObstructionGroup,
    OpenCylinderGroup,
    OrientedBoxGroup,
    SphereGroup,
    TriangleGroup,
)
from ._common import as_vec3

__all__ = [
    # Re-exported core primitives
    "ObstructionGroup",
    "CylinderGroup",
    "OpenCylinderGroup",
    "BoxGroup",
    "SphereGroup",
    "OrientedBoxGroup",
    "TriangleGroup",
    # High-level factories
    "cylinder",
    "open_cylinder",
    "box",
    "sphere",
]


def _as_vec3(value, name: str) -> list[float]:
    """Coerce to shape ``(3,)``, then a plain list (these factories build list-of-lists)."""
    return as_vec3(value, name).tolist()


# Single-primitive factories


def cylinder(
    *,
    p1: Sequence[float],
    p2: Sequence[float],
    r: float,
) -> CylinderGroup:
    """Build a single closed cylinder as a ``CylinderGroup`` of size one.

    Args:
        p1: One endpoint of the cylinder axis, shape (3,).
        p2: The other endpoint, shape (3,).
        r: Cylinder radius in metres.
    """
    return CylinderGroup(
        p1=[_as_vec3(p1, "p1")],
        p2=[_as_vec3(p2, "p2")],
        r=[float(r)],
    )


def open_cylinder(
    *,
    p1: Sequence[float],
    p2: Sequence[float],
    r: float,
) -> OpenCylinderGroup:
    """Build a single open cylinder (no end caps) as a size-one group.

    Args:
        p1: One endpoint of the cylinder axis, shape (3,).
        p2: The other endpoint, shape (3,).
        r: Cylinder radius in metres.
    """
    return OpenCylinderGroup(
        p1=[_as_vec3(p1, "p1")],
        p2=[_as_vec3(p2, "p2")],
        r=[float(r)],
    )


def box(
    *,
    p1: Sequence[float],
    p2: Sequence[float],
) -> BoxGroup:
    """Build a single axis-aligned box as a ``BoxGroup`` of size one.

    Args:
        p1: One corner of the box, shape (3,).
        p2: The diagonally opposite corner, shape (3,).
    """
    return BoxGroup(
        p1=[_as_vec3(p1, "p1")],
        p2=[_as_vec3(p2, "p2")],
    )


def sphere(
    *,
    center: Sequence[float],
    r: float,
) -> SphereGroup:
    """Build a single sphere as a ``SphereGroup`` of size one.

    Args:
        center: Sphere centre, shape (3,).
        r: Sphere radius in metres.
    """
    return SphereGroup(
        centers=[_as_vec3(center, "center")],
        radii=[float(r)],
    )
