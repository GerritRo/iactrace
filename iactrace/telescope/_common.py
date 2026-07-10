"""Shared input-shape helpers for the telescope factory modules."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array


def as_vec3(value, name: str) -> Array:
    """Coerce ``value`` to a shape-``(3,)`` array, raising with ``name`` on mismatch."""
    arr = jnp.asarray(value)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def as_aspheric_row(coeffs: Sequence[float] | None) -> Array:
    """Aspheric coefficient row; ``None`` becomes an empty ``(0,)`` array."""
    if coeffs is None:
        return jnp.zeros((0,))
    return jnp.asarray(coeffs)
