"""R(theta, lambda) response curves and n(lambda) refractive indices."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...core.refractive_index import (
    ConstantIndex,
    RefractiveIndex,
    SellmeierIndex,
    TabulatedIndex,
)
from ...core.responses import ResponseCurve, TabulatedResponse
from ..schemas import (
    RefractiveIndexSchema,
    SellmeierIndexSchema,
    TabulatedCurveSchema,
    TabulatedIndexSchema,
)


def _curves_equal(
    a: TabulatedCurveSchema,
    b: TabulatedCurveSchema,
) -> bool:
    """Structural equality of two tabulated curve schemas."""
    return (
        a.angles_deg == b.angles_deg
        and a.values == b.values
        and a.wavelengths_nm == b.wavelengths_nm
    )


def _build_curve_for_bucket(
    curves: list[TabulatedCurveSchema | None],
    n_elements: int,
) -> ResponseCurve | None:
    """Resolve a list of per-element curve schemas into a single response curve.

    All None -> None (caller's default physics applies).
    One distinct curve -> broadcast across all elements.
    Coated mixed with uncoated, or several distinct curves -> ValueError.
    """
    distinct: list[TabulatedCurveSchema] = []
    for c in curves:
        if c is None:
            continue
        if not any(_curves_equal(c, d) for d in distinct):
            distinct.append(c)

    if not distinct:
        return None
    if any(c is None for c in curves):
        raise ValueError(
            "Elements grouped at the same stage with the same aperture "
            "must either all define a response curve or all omit it; mixing "
            "curved and flat elements would silently apply one element's "
            "curve to the rest. Split them across stages or harmonize their "
            "`*_curve` fields."
        )
    if len(distinct) > 1:
        raise ValueError(
            "Elements grouped at the same stage with the same aperture "
            "must resolve to a single response curve, but multiple distinct "
            "curves were found; broadcasting one would silently apply it to "
            "the rest. Split them across stages or harmonize their "
            "`*_curve` fields."
        )

    curve = distinct[0]
    return TabulatedResponse.from_degrees(
        angles_deg=curve.angles_deg,
        values=curve.values,
        n_elements=n_elements,
        wavelengths=curve.wavelengths_nm,
    )


def _indices_equal(a: RefractiveIndexSchema, b: RefractiveIndexSchema) -> bool:
    """Structural equality of two refractive-index schemas."""
    return a.model_dump() == b.model_dump()


def _build_index_for_bucket(
    schemas: list[float | RefractiveIndexSchema],
    n_elements: int,
) -> RefractiveIndex | Array:
    """Resolve per-element index fields into one bucket-wide index argument.

    Plain numbers stay per-element and come back as an (N,) array (a
    non-dispersive bucket may mix values freely). A dispersive model is
    bucket-wide, so mixing a model with numbers, or two distinct models,
    raises rather than silently applying one element's dispersion to the rest.
    """
    models = [s for s in schemas if not isinstance(s, float | int)]
    if not models:
        return jnp.asarray([float(cast("float", s)) for s in schemas])
    if len(models) != len(schemas):
        raise ValueError(
            "Elements grouped at the same stage with the same aperture must "
            "either all define a dispersive `index` model or all give a plain "
            "number; mixing them would silently apply one element's dispersion "
            "to the rest. Split them across stages."
        )
    first = models[0]
    if any(not _indices_equal(first, s) for s in models[1:]):
        raise ValueError(
            "Elements grouped at the same stage with the same aperture must "
            "resolve to a single `index` model, but multiple distinct models "
            "were found. Split them across stages or harmonise their `index`."
        )
    if isinstance(first, SellmeierIndexSchema):
        b = jnp.broadcast_to(jnp.asarray(first.b), (n_elements, len(first.b)))
        c = jnp.broadcast_to(jnp.asarray(first.c), (n_elements, len(first.c)))
        return SellmeierIndex(b=b, c=c)
    return TabulatedIndex.from_table(first.wavelengths_nm, first.n, n_elements)


def _curve_to_schema(
    curve: ResponseCurve | None,
) -> TabulatedCurveSchema | None:
    """Project a ResponseCurve to a serialisable curve, or None if trivial.

    None and ConstantResponse round-trip as None so
    existing YAMLs stay byte-identical. A TabulatedResponse
    emits the inline {type: table, ...} form -- an angle-only 1-D
    values list when the curve has no wavelength axis (byte-identical
    to before), or an (angle, wavelength) grid with wavelengths_nm
    when it does. The YAML schema holds one shared curve per template, so a
    per-element curve (rows that differ across elements) raises
    ValueError rather than silently serialising only the first
    element's row.

    Raises
    ------
    ValueError
        If curve is a per-element
        TabulatedResponse whose rows are not all equal.
    """
    if isinstance(curve, TabulatedResponse):
        value_rows = np.asarray(curve.values)  # (N, Kc, Kw)
        # YAML expresses one shared curve per template. A per-element
        # curve (rows differ) cannot be represented; fail loudly rather
        # than silently serialising only element 0's row. Mirrors the
        # loader guard in _build_curve_for_bucket.
        if value_rows.shape[0] > 1 and not np.allclose(value_rows, value_rows[0]):
            raise ValueError(
                "Cannot serialise a per-element TabulatedResponse to YAML: "
                "all elements in a group must share one curve. Split them "
                "across groups, or harmonise their rows before saving."
            )
        grid = value_rows[0]  # (Kc, Kw)
        cos_table = np.asarray(curve.cos_table)
        order = np.argsort(-cos_table)  # cos descending -> angles ascending
        angles_deg = [float(x) for x in np.degrees(np.arccos(cos_table[order]))]
        if curve.wl_table.shape[0] == 1:
            # Angle-only curve: emit the 1-D form (byte-identical to before).
            values = [float(x) for x in grid[order, 0]]
            return TabulatedCurveSchema(angles_deg=angles_deg, values=values)
        # (angle, wavelength) grid; wl_table is already ascending.
        wl_table = np.asarray(curve.wl_table)
        values2d = [[float(x) for x in grid[a, :]] for a in order]
        return TabulatedCurveSchema(
            angles_deg=angles_deg,
            values=values2d,
            wavelengths_nm=[float(w) for w in wl_table],
        )
    return None


def _curve_schema_to_key(
    schema: TabulatedCurveSchema | None,
) -> tuple | None:
    """Hashable key used by mirrors_to_schemas to dedup templates."""
    if schema is None:
        return None
    if schema.wavelengths_nm is None:
        values_key: tuple = tuple(cast("list[float]", schema.values))
        wl_key: tuple | None = None
    else:
        rows = cast("list[list[float]]", schema.values)
        values_key = tuple(tuple(row) for row in rows)
        wl_key = tuple(schema.wavelengths_nm)
    return ("table", tuple(schema.angles_deg), values_key, wl_key)


def _index_to_schema(index: RefractiveIndex, i: int) -> float | RefractiveIndexSchema:
    """Serialise element i of an index model to the YAML index field.

    The one field takes either form, mirroring the Python argument: a
    ConstantIndex emits the plain number, a dispersive model emits
    its schema.
    """
    if isinstance(index, ConstantIndex):
        return float(index.values[i])
    if isinstance(index, SellmeierIndex):
        return SellmeierIndexSchema(
            b=[float(x) for x in np.asarray(index.b[i])],
            c=[float(x) for x in np.asarray(index.c[i])],
        )
    if isinstance(index, TabulatedIndex):
        return TabulatedIndexSchema(
            wavelengths_nm=[float(x) for x in np.asarray(index.wavelengths)],
            n=[float(x) for x in np.asarray(index.n_values[i])],
        )
    raise ValueError(f"Cannot serialise refractive index of type {type(index).__name__} to YAML.")
