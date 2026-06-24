from __future__ import annotations

import jax.numpy as jnp
from jax import Array

SQRT3: float = 1.7320508075688772
SQRT3_2: float = 0.8660254037844386  # sqrt(3)/2
SQRT3_3: float = 0.5773502691896257  # 1/sqrt(3)


def _rotate(x: Array, y: Array, angle: float | Array) -> tuple[Array, Array]:
    """Rotate 2D coordinates by angle."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return c * x - s * y, s * x + c * y


def _cartesian_to_axial(x: Array, y: Array, size: float) -> tuple[Array, Array]:
    """Cartesian to axial hex coordinates (pointy-top)."""
    return (SQRT3_3 * x - y / 3) / size, (2 * y / 3) / size


def _axial_to_cartesian(q: Array, r: Array, size: float) -> tuple[Array, Array]:
    """Axial to Cartesian hex coordinates (pointy-top)."""
    return size * SQRT3 * (q + r / 2), size * 1.5 * r


def _axial_round(q: Array, r: Array) -> tuple[Array, Array]:
    """Round fractional axial coordinates to nearest hex center."""
    s = -q - r
    qi, ri, si = jnp.round(q), jnp.round(r), jnp.round(s)
    dq, dr, ds = jnp.abs(qi - q), jnp.abs(ri - r), jnp.abs(si - s)
    qi = jnp.where((dq > dr) & (dq > ds), -ri - si, qi)
    ri = jnp.where((dr > dq) & (dr > ds), -qi - si, ri)
    return qi, ri


def _hex_norm(x: Array, y: Array, inradius: float) -> Array:
    """Hexagonal norm: 0 at center, 1 at boundary (pointy-top)."""
    return jnp.maximum(jnp.abs(x), 0.5 * jnp.abs(x) + SQRT3_2 * jnp.abs(y)) / inradius


def _detect_hex_grid(centers: Array) -> tuple[Array, Array, Array]:
    """Detect hex size, rotation, and offset from center positions."""
    centers = jnp.asarray(centers)
    n = len(centers)

    diff = centers[:, None] - centers[None, :]
    dist_sq = jnp.sum(diff**2, axis=2)
    dist_sq = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist_sq)
    min_dist = jnp.sqrt(jnp.min(dist_sq))

    idx = jnp.argmin(dist_sq)
    vec = diff[idx // n, idx % n]
    angle = jnp.mod(jnp.arctan2(vec[1], vec[0]), jnp.pi / 3)

    offset = centers[jnp.argmin(jnp.sum(centers**2, axis=1))]

    return min_dist / SQRT3, angle, offset


def _build_lookup_table(
    centers: Array, hex_size: float, rotation: float, offset: Array
) -> tuple[Array, int, int]:
    """Build axial coordinate lookup table from hex centers."""
    x = centers[:, 0] - offset[0]
    y = centers[:, 1] - offset[1]
    x_rot, y_rot = _rotate(x, y, -rotation)

    q, r = _cartesian_to_axial(x_rot, y_rot, hex_size)
    qi = jnp.round(q).astype(jnp.int32)
    ri = jnp.round(r).astype(jnp.int32)

    q_min, q_max = int(qi.min()), int(qi.max())
    r_min, r_max = int(ri.min()), int(ri.max())

    table = jnp.full((q_max - q_min + 1, r_max - r_min + 1), -1, dtype=jnp.int32)
    table = table.at[qi - jnp.int32(q_min), ri - jnp.int32(r_min)].set(
        jnp.arange(len(centers), dtype=jnp.int32)
    )

    return table, q_min, r_min
