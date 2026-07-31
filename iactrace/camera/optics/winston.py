from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .polygonal import PolygonalCone

_T_FLOOR = 1e-6  # spurious-hit rejection floor, scaled by a2


def cpc_wall_tilt(
    exit_apothem: float, entrance_apothem: float, length: float
) -> tuple[float, float]:
    """(sin, cos) of the CPC wall tilt for a cone whose exit-rim apothem is
    a2 and whose +u wall passes through the entrance (a1, length).

    Raises:
        ValueError: if (a2, a1, length) is not a realizable CPC.
    """
    a2, a1, length = float(exit_apothem), float(entrance_apothem), float(length)
    if not 0.0 < a2 < a1:
        raise ValueError(f"require 0 < exit_apothem < entrance_apothem, got a2={a2}, a1={a1}")
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    b = a2 - a1
    r2 = length * length + b * b
    d = math.hypot(a1 + a2, length) - 2.0 * a2
    disc = r2 - d * d
    if disc < 0.0:
        raise ValueError(f"(exit={a2}, entrance={a1}, length={length}) is not a realizable ")
    sq = math.sqrt(disc)
    s = (b * d + length * sq) / r2
    c = (length * d - b * sq) / r2
    if not (0.0 < s < 1.0 and c > 0.0):
        raise ValueError(
            f"(exit={a2}, entrance={a1}, length={length}) yields a non-physical "
            f"wall tilt (sin={s:.4g}, cos={c:.4g})."
        )
    return s, c


def cpc_ideal_wall_tilt(exit_apothem: float, entrance_apothem: float) -> tuple[float, float]:
    """(sin, cos) of the wall tilt for the untruncated (ideal) CPC.

    For the full cone, ``entrance_apothem`` *is* the full CPC entry a1, which
    fixes the wall tilt directly: ``sin(theta) = a2 / a1``.
    """
    s = exit_apothem / entrance_apothem
    c = math.sqrt(1.0 - s * s)
    return s, c


def cpc_full_length(exit_apothem: float, s: float, c: float) -> float:
    """Full (untruncated) CPC length for exit apothem a2 and wall tilt (s, c).

    The full entry is a1 = a2 / s and the full depth is (a1 + a2)*c/s.
    """
    a1 = exit_apothem / s
    return (a1 + exit_apothem) * c / s


def profile_apothem(z: Array, exit_apothem: float, s: float, c: float) -> Array:
    """Wall apothem R(z) in CPC coords (R(0)=a2 exit, R(L)=a1 entrance).

    Solves the meridian parabola Q(u, z) = 0 for the physical +u root,
    given the exit apothem and the wall tilt (s, c).
    """
    a2 = exit_apothem
    k = a2 * (2.0 + s)
    A = c * c
    B = 2.0 * (a2 + s * c * z + s * k)
    C = a2**2 + z**2 - (c * z + k) ** 2
    # Grad-safe sqrt
    raw = B * B - 4.0 * A * C
    pos = raw > 0.0
    sq = jnp.where(pos, jnp.sqrt(jnp.where(pos, raw, 1.0)), 0.0)
    return (-B + sq) / (2.0 * A)


def _wall_t(o: Array, d: Array, n: Array, a2: float, s: float, c: float, k: float) -> Array:
    """Smallest forward ray parameter hitting wall n (inf if none)."""
    p = d[0] * n[0] + d[1] * n[1]
    q = d[2]
    u0 = o[0] * n[0] + o[1] * n[1]
    z0 = o[2]
    m0 = -s * u0 + c * z0 + k
    mp = -s * p + c * q
    A = (p * c + q * s) ** 2
    B = 2.0 * ((u0 + a2) * p + z0 * q - m0 * mp)
    C = (u0 + a2) ** 2 + z0**2 - m0**2
    disc = B * B - 4.0 * A * C
    # Grad-safe roots:
    pos = disc > 0.0
    sq = jnp.where(pos, jnp.sqrt(jnp.where(pos, disc, 1.0)), 0.0)
    safe_A = jnp.where(jnp.abs(A) < 1e-14, 1.0, A)
    safe_B = jnp.where(jnp.abs(B) > 1e-30, B, 1.0)
    # Inside the cavity C < 0, so the positive root is (-B + sq)/(2A).
    t_quad = (-B + sq) / (2.0 * safe_A)
    t_lin = jnp.where(jnp.abs(B) > 1e-30, -C / safe_B, jnp.inf)
    t = jnp.where(jnp.abs(A) < 1e-14, t_lin, t_quad)
    # Reject backward / spurious near-zero hits with a floor that scales with the
    # cone size.
    bad = (disc < 0) | (t <= _T_FLOOR * a2) | ~jnp.isfinite(t)
    return jnp.where(bad, jnp.inf, t)


def _wall_normal(P: Array, n: Array, a2: float, s: float, c: float, k: float) -> Array:
    """Unit normal of wall n at point P."""
    u = P[0] * n[0] + P[1] * n[1]
    z = P[2]
    m = -s * u + c * z + k
    dq_du = 2.0 * (u + a2) + 2.0 * m * s
    dq_dz = 2.0 * z - 2.0 * m * c
    nrm = jnp.array([dq_du * n[0], dq_du * n[1], dq_dz])
    # Grad-safe normalize:
    norm_sq = nrm @ nrm
    return nrm / jnp.sqrt(jnp.where(norm_sq > 0.0, norm_sq, 1.0))


class WinstonCone(PolygonalCone):
    """Polygonal CPC (Winston cone) light guide.

    Defined entirely by its physical dimensions; exit apothem, entrance apothem
    and length. The parabolic-wall tilt ``(s, c)`` that fixes the cone is
    computed from them at construction (see :func:`cpc_wall_tilt`). The cone
    answers the per-facet meridian-parabola hit (:meth:`_nearest_hit`); the
    bounce loop is owned by the shared :func:`~iactrace.camera.optics.polygonal.trace_chain`.

    Args:
        n_sides: Number of facets (6 = hexagonal, 4 = square, ...).
        entrance_apothem: Entrance inradius ``a1``; the apothem **at the entrance
            plane** ``z = length``. For a truncated cone this is the actual
            (truncated) entry.
        exit_apothem: Exit aperture inradius ``a2``.
        length: Physical depth. ``None`` builds the full (untruncated) CPC and
            derives the length from ``a1``/``a2``; a value truncates the cone (then
            ``entrance_apothem`` is the entry at that depth).
        reflectivity: Per-bounce wall reflectivity (scalar).
        max_bounces: Maximum reflections traced before a ray is absorbed.
        orientation_deg: Rotation of the polygon about the optical axis.
    """

    n_sides: int = eqx.field(static=True)
    exit_apothem: float = eqx.field(static=True)
    entrance_apothem: float = eqx.field(static=True)
    reflectivity: float = eqx.field(static=True)
    max_bounces: int = eqx.field(static=True)
    orientation: float = eqx.field(static=True)  # radians
    length: float = eqx.field(static=True)
    s: float = eqx.field(static=True)  # sin(wall tilt), fixed at construction
    c: float = eqx.field(static=True)  # cos(wall tilt), fixed at construction

    def __init__(
        self,
        n_sides: int,
        entrance_apothem: float,
        exit_apothem: float,
        length: float | None = None,
        reflectivity: float = 0.9,
        max_bounces: int = 10,
        orientation_deg: float = 0.0,
    ) -> None:
        if not 0.0 < exit_apothem < entrance_apothem:
            raise ValueError(
                "require 0 < exit_apothem < entrance_apothem, got "
                f"exit_apothem={exit_apothem}, entrance_apothem={entrance_apothem}"
            )
        self.n_sides = int(n_sides)
        self.exit_apothem = float(exit_apothem)
        self.entrance_apothem = float(entrance_apothem)
        self.reflectivity = float(reflectivity)
        self.max_bounces = int(max_bounces)
        self.orientation = math.radians(float(orientation_deg))

        if length is None:
            s, c = cpc_ideal_wall_tilt(self.exit_apothem, self.entrance_apothem)
            self.length = float(cpc_full_length(self.exit_apothem, s, c))
        else:
            # Truncated cone: entrance_apothem is the physical entry at z = length;
            # the wall tilt (hence the parabola) follows from the three dimensions.
            s, c = cpc_wall_tilt(self.exit_apothem, self.entrance_apothem, length)
            full_length = cpc_full_length(self.exit_apothem, s, c)
            if float(length) > full_length + 1e-9:
                raise ValueError(
                    f"length={length} exceeds the full CPC length "
                    f"{full_length:.6g}; truncation only shortens the cone."
                )
            self.length = float(length)
        self.s = float(s)
        self.c = float(c)

    @property
    def k(self) -> float:
        """Meridian offset ``a2 * (2 + s)`` of the wall parabola."""
        return self.exit_apothem * (2.0 + self.s)

    def _nearest_hit(self, o: Array, d: Array) -> tuple[Array, Array]:
        t_all = jax.vmap(lambda nh: _wall_t(o, d, nh, self.exit_apothem, self.s, self.c, self.k))(
            self.n_hats
        )
        kbest = jnp.argmin(t_all)
        t = t_all[kbest]
        p = o + jnp.where(jnp.isfinite(t), t, 0.0) * d
        normal = _wall_normal(p, self.n_hats[kbest], self.exit_apothem, self.s, self.c, self.k)
        return t, normal

    def _meridian(self) -> tuple[Array, Array]:
        # entrance at z=0 (apothem a1) down to the exit at z=-length (apothem a2)
        z_chain = jnp.linspace(0.0, -self.length, self._N_SLICES)
        apothem = profile_apothem(self.length + z_chain, self.exit_apothem, self.s, self.c)
        return z_chain, apothem
