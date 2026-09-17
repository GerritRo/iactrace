"""Surface shapes: aspheric, Zernike, and sums of the two."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...core.optics import OpticalElementGroup
from ...core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from ..schemas import AsphericSurfaceSchema, ZernikeSurfaceSchema
from ._common import _pad_aspherics, _strip_trailing_zeros, _to_float_list


def _surface_list(spec) -> list:
    """Normalise a surface spec (a single shape or a list) to a list of shapes."""
    return list(spec) if isinstance(spec, list) else [spec]


def _split_surface(
    spec,
) -> tuple[AsphericSurfaceSchema | None, ZernikeSurfaceSchema | None]:
    """Split a surface spec into its (aspheric, zernike) shapes.

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

    Reuses _split_surface -- the same aspheric/zernike decomposition
    every mirror and lens surface goes through -- so any surface an optical
    element can describe (a bare aspheric shape, a bare zernike shape, or
    their sum) is buildable from a single spec. The N-wide counterpart lives
    in _build_mirror_group / _build_aspheric_disk_lens_group
    (batched over a bucket); this is the N=1 case, used by
    _pmt_from_schema.
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
    zernike = ZernikeSurfaceGroup(
        coeffs=jnp.asarray([zern.coeffs]), r_norm=jnp.asarray([zern.r_norm])
    )
    if asph is None:
        return zernike
    return SumSurfaceGroup([aspheric, zernike])


def _build_zernike_for_bucket(
    schemas: list[ZernikeSurfaceSchema | None],
) -> ZernikeSurfaceGroup | None:
    """Reassemble one group's Zernike term from per-element schemas, or None.
    All None -> None (no figure error). Otherwise every element's
    coefficients are padded to a common width and stacked; elements without a
    zernike block contribute zero coefficients (and a placeholder r_norm
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
        coeffs=jnp.asarray(coeffs),
        r_norm=jnp.asarray(r_norms),
    )


def _compose_surface(
    group: OpticalElementGroup,
    zernike_schemas: list[ZernikeSurfaceSchema | None],
    *,
    has_aspheric: bool,
) -> OpticalElementGroup:
    """Replace the group's built aspheric surface with the composed surface.

    The group is always built with an AsphericSurfaceGroup (flat when
    the spec has no aspheric shape). Given the per-element Zernike shapes and
    whether the bucket has an aspheric shape at all:

    - no Zernike -> keep the aspheric surface (bare asphere);
    - Zernike + aspheric -> SumSurfaceGroup([asphere, zernike]);
    - Zernike only -> a standalone ZernikeSurfaceGroup (the flat
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


def _surface_components(
    surface,
) -> tuple[AsphericSurfaceGroup | None, ZernikeSurfaceGroup | None]:
    """Split a surface into its aspheric and Zernike parts for serialization.
    Accepts a bare AsphericSurfaceGroup, a standalone
    ZernikeSurfaceGroup, or a SumSurfaceGroup composing one of
    each. Either part may be None. Raises if the surface contains anything
    else, more than one of either type, or a non-zero decenter on the composite
    or the Zernike term (the flat per-element schema keeps the decenter on the
    asphere only).
    """
    if isinstance(surface, AsphericSurfaceGroup):
        return surface, None
    if isinstance(surface, ZernikeSurfaceGroup):
        if not np.allclose(np.asarray(surface.offsets), 0.0):
            raise ValueError("cannot serialise a Zernike surface with a non-zero decenter")
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
            raise ValueError("cannot serialise a Zernike term with a non-zero decenter")
        return asph, zern
    raise ValueError(f"cannot serialise surface type {type(surface).__name__}")


def _zernike_to_schema(zernike: ZernikeSurfaceGroup | None, i: int) -> ZernikeSurfaceSchema | None:
    """Project element i of a Zernike term to a schema, or None.
    Elements whose coefficients are all zero round-trip as None so default
    (figure-error-free) elements stay clean in the YAML.
    """
    if zernike is None:
        return None
    coeffs = _strip_trailing_zeros(_to_float_list(zernike.coeffs[i]))
    if not coeffs:
        return None
    return ZernikeSurfaceSchema(coeffs=coeffs, r_norm=float(zernike.r_norm[i]))


def _surface_to_spec(asph, zern, i: int):
    """Serialise element i's surface into a spec: one shape, or a summed list.

    An aspheric shape comes first (it supplies the intersection guess); a
    non-trivial Zernike term follows. A standalone Zernike surface serialises as
    a single zernike shape.
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
    """Return (asphere, zernike, offsets) for a group's surface.
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
