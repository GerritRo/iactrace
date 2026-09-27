"""Surface-roughness scattering models."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from ...core.bsdf import BSDF, DoubleGaussianBSDF, GaussianBSDF
from ..schemas import BSDFSchema, DoubleGaussianBSDFSchema, GaussianBSDFSchema


class _BsdfSpec(NamedTuple):
    """Bidirectional spec for one BSDF type; see _ConcentratorSpec.

    build assembles a whole-bucket domain BSDF from a list of per-element
    schemas (already known to be homogeneous and non-empty); to_schema
    projects element i of a domain BSDF back to a schema, or None for
    a trivially-zero element.
    """

    type_name: str
    schema: type
    group: type
    build: Callable[[list], BSDF | None]
    # Each entry's to_schema only ever accepts that entry's own BSDF subclass
    # (the driver looks it up by type(bsdf) first), narrower than a plain
    # Callable[[BSDF, int], ...] would allow; typed loosely here for that reason.
    to_schema: Callable[..., BSDFSchema | None]


def _build_gaussian_bsdf(schemas: list[GaussianBSDFSchema | None]) -> GaussianBSDF | None:
    scale = jnp.asarray([s.scale if s is not None else 0.0 for s in schemas])
    if bool(jnp.all(scale == 0)):
        return None
    return GaussianBSDF(scale=scale)


def _gaussian_bsdf_to_schema(bsdf: GaussianBSDF, i: int) -> GaussianBSDFSchema | None:
    scale = float(bsdf.scale[i])
    return None if scale == 0.0 else GaussianBSDFSchema(scale=scale)


def _build_double_gaussian_bsdf(
    schemas: list[DoubleGaussianBSDFSchema | None],
) -> DoubleGaussianBSDF:
    def _col(attr: str) -> Array:
        return jnp.asarray([getattr(s, attr) if s is not None else 0.0 for s in schemas])

    return DoubleGaussianBSDF(
        scale_narrow=_col("scale_narrow"),
        scale_wide=_col("scale_wide"),
        mix_weight=_col("mix_weight"),
    )


def _double_gaussian_bsdf_to_schema(bsdf: DoubleGaussianBSDF, i: int) -> DoubleGaussianBSDFSchema:
    return DoubleGaussianBSDFSchema(
        scale_narrow=float(bsdf.scale_narrow[i]),
        scale_wide=float(bsdf.scale_wide[i]),
        mix_weight=float(bsdf.mix_weight[i]),
    )


# The single source of truth for BSDF round-tripping; see _CONCENTRATOR_SPECS.
# Adding a BSDF model is one entry here plus a schema variant in
# iactrace.io.schemas.
_BSDF_SPECS: tuple[_BsdfSpec, ...] = (
    _BsdfSpec(
        "gaussian", GaussianBSDFSchema, GaussianBSDF, _build_gaussian_bsdf, _gaussian_bsdf_to_schema
    ),
    _BsdfSpec(
        "double_gaussian",
        DoubleGaussianBSDFSchema,
        DoubleGaussianBSDF,
        _build_double_gaussian_bsdf,
        _double_gaussian_bsdf_to_schema,
    ),
)
_BSDF_SPEC_BY_GROUP: dict[type, _BsdfSpec] = {s.group: s for s in _BSDF_SPECS}
_BSDF_SPEC_BY_TYPE: dict[str, _BsdfSpec] = {s.type_name: s for s in _BSDF_SPECS}


def _build_bsdf_for_bucket(
    schemas: list[BSDFSchema | None],
) -> BSDF | None:
    """Reassemble one group's BSDF from per-element schemas; see _BSDF_SPECS.

    All None -> None (perfect specular). Otherwise every element
    that declares a BSDF must share the same type; per-element
    parameters are stacked into the model's arrays, and elements without
    a BSDF default to zero (specular for that element). Mixed types
    raise ValueError, mirroring the per-bucket curve guard in
    _build_curve_for_bucket.
    """
    present = [s for s in schemas if s is not None]
    if not present:
        return None

    types = {s.type for s in present}
    if len(types) > 1:
        raise ValueError(
            "Mirrors grouped at the same stage with the same aperture must "
            f"share a single BSDF type; got {sorted(types)}. Split them "
            "across stages, or harmonise their `bsdf.type`."
        )

    spec = _BSDF_SPEC_BY_TYPE.get(present[0].type)
    if spec is None:  # pragma: no cover - unreachable while the union is exhaustive
        raise ValueError(
            f"Unhandled BSDF schema type {present[0].type!r}; add a "
            "_BsdfSpec entry in iactrace.io.adapters."
        )
    return spec.build(schemas)


def _bsdf_to_schema(bsdf: BSDF | None, i: int) -> BSDFSchema | None:
    """Project element i of a group BSDF to a serialisable schema; see _BSDF_SPECS.

    None and an all-zero GaussianBSDF
    element round-trip as None so default (specular) mirrors stay
    clean in the YAML. Unhandled BSDF subclasses raise rather than being
    silently dropped to a partial form.
    """
    if bsdf is None:
        return None
    spec = _BSDF_SPEC_BY_GROUP.get(type(bsdf))
    if spec is None:
        raise ValueError(
            f"BSDF type {type(bsdf).__name__} cannot be serialised to "
            "YAML; add a schema variant in iactrace.io.schemas and a "
            "_BsdfSpec entry in iactrace.io.adapters."
        )
    return spec.to_schema(bsdf, i)
