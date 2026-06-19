from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.interactions import reflect
from ..core.ray_bundle import RayBundle
from .concentrator import Concentrator

_NUDGE = 1e-5     # off-wall step, scaled by a2
_T_FLOOR = 1e-6   # spurious-hit rejection floor


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
        raise ValueError(
            f"require 0 < exit_apothem < entrance_apothem, got a2={a2}, a1={a1}"
        )
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    b = a2 - a1
    r2 = length * length + b * b
    d = math.hypot(a1 + a2, length) - 2.0 * a2
    disc = r2 - d * d
    if disc < 0.0:
        raise ValueError(
            f"(exit={a2}, entrance={a1}, length={length}) is not a realizable "
        )
    sq = math.sqrt(disc)
    s = (b * d + length * sq) / r2
    c = (length * d - b * sq) / r2
    if not (0.0 < s < 1.0 and c > 0.0):
        raise ValueError(
            f"(exit={a2}, entrance={a1}, length={length}) yields a non-physical "
            f"wall tilt (sin={s:.4g}, cos={c:.4g})."
        )
    return s, c


def cpc_full_length(exit_apothem: float, s: float, c: float) -> float:
    """Full (untruncated) CPC length for exit apothem a2 and wall tilt (s, c).

    The full entry is a1 = a2 / s and the full depth is (a1 + a2)·c/s.
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
    disc = jnp.maximum(B * B - 4.0 * A * C, 0.0)
    return (-B + jnp.sqrt(disc)) / (2.0 * A)


def _wall_t(o: Array, d: Array, n: Array, a2: float, s: float, c: float,
            k: float) -> Array:
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
    # cone size (tracks the a2-scaled nudge, stays above float32 rounding noise).
    bad = (disc < 0) | (t <= _T_FLOOR * a2) | ~jnp.isfinite(t)
    return jnp.where(bad, jnp.inf, t)


def _wall_normal(P: Array, n: Array, a2: float, s: float, c: float,
                 k: float) -> Array:
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


def _single_step(o, d, value, path, done, n_hats, a2, s, c, k, length, refl):
    """One reflection event for one ray (frozen once ``done``)."""
    t_all = jax.vmap(lambda n: _wall_t(o, d, n, a2, s, c, k))(n_hats)
    kbest = jnp.argmin(t_all)
    t_wall = t_all[kbest]

    dz = d[2]
    safe_dz = jnp.where(dz != 0.0, dz, 1.0)
    t_exit = jnp.where(dz < 0, (0.0 - o[2]) / safe_dz, jnp.inf)
    t_ent = jnp.where(dz > 0, (length - o[2]) / safe_dz, jnp.inf)

    is_exit = jnp.isfinite(t_exit) & (t_exit <= t_wall) & (t_exit <= t_ent)
    is_wall = jnp.isfinite(t_wall) & (t_wall < t_exit) & (t_wall <= t_ent)
    is_lost = (~is_exit) & (~is_wall)

    tw = jnp.where(jnp.isfinite(t_wall), t_wall, 0.0)
    pw = o + tw * d
    nrm = _wall_normal(pw, n_hats[kbest], a2, s, c, k)
    dw, _ = reflect(d, nrm)
    # Step off the wall along the reflected ray so the next intersection test
    # sees this wall behind it (t < 0). The nudge lies on the outgoing ray, so
    # it is added back to the optical path and the geometry stays exact.
    nudge = _NUDGE * a2
    pw = pw + nudge * dw
    te = jnp.where(jnp.isfinite(t_exit), t_exit, 0.0)

    o_new = jnp.where(is_wall, pw, jnp.where(is_exit, o + te * d, o))
    d_new = jnp.where(is_wall, dw, d)
    value_new = jnp.where(is_lost, 0.0, jnp.where(is_wall, value * refl, value))
    path_new = path + jnp.where(is_wall, tw + nudge, jnp.where(is_exit, te, 0.0))
    done_new = done | is_exit | is_lost

    live = ~done
    return (
        jnp.where(live, o_new, o),
        jnp.where(live, d_new, d),
        jnp.where(live, value_new, value),
        jnp.where(live, path_new, path),
        jnp.where(live, done_new, done),
    )



def trace(origins, directions, n_hats, exit_apothem, s, c, length, reflectivity,
          max_bounces):
    """Trace rays through the cone in CPC coords (exit at z=0, entrance at z=length).

    Returns ``(exit_origins, exit_directions, value_factor, path_added)``;
    ``value_factor`` is ``reflectivity**bounces`` for transmitted rays and ``0``
    for rays that miss the entry, leave back through the entrance, or never exit
    within ``max_bounces``.
    """
    a2 = exit_apothem
    k = a2 * (2.0 + s)
    n = origins.shape[0]

    # Rays entering outside the cone entry are lost
    ent_apothem = profile_apothem(jnp.asarray(length), a2, s, c)
    u_all = origins[:, :2] @ n_hats.T
    inside = jnp.all(u_all <= ent_apothem + 1e-9, axis=1)

    carry = (
        origins, directions,
        jnp.where(inside, 1.0, 0.0),
        jnp.zeros(n),
        ~inside,
    )

    step_fn: Callable[..., tuple[Array, Array, Array, Array, Array]] = jax.vmap(
        _single_step,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )

    def step(carry, _):
        out = step_fn(*carry, n_hats, a2, s, c, k, length, reflectivity)
        return out, None

    (o, d, value, path, done), _ = jax.lax.scan(
        step, carry, None, length=max_bounces + 1,
    )
    # Rays still bouncing at the end are absorbed.
    value = jnp.where(done, value, 0.0)
    return o, d, value, path

# WinstonCone concentrator


class WinstonCone(Concentrator):
    """Polygonal CPC (Winston cone) light guide with full ray tracing.

    Defined entirely by its physical dimensions — exit apothem, entrance apothem
    and length. The parabolic-wall tilt that fixes the cone is computed directly
    from them (see :func:`cpc_wall_tilt`).

    Args:
        n_sides: Number of facets (6 = hexagonal, 4 = square, ...).
        entrance_apothem: Entrance inradius ``a1`` — the apothem **at the entrance
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
    orientation: float = eqx.field(static=True) # radians
    length: float = eqx.field(static=True)

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
            # Untruncated cone: entrance_apothem is the full CPC entry a1, which
            # fixes the wall tilt directly (sin = a2 / a1).
            s = self.exit_apothem / self.entrance_apothem
            c = math.sqrt(1.0 - s * s)
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

    def _wall_tilt(self) -> tuple[float, float]:
        """(sin, cos) of the wall tilt, from the cone's physical dimensions."""
        return cpc_wall_tilt(self.exit_apothem, self.entrance_apothem, self.length)

    def _wall_normals(self) -> Array:
        a = self.orientation + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides
        return jnp.stack([jnp.cos(a), jnp.sin(a)], axis=-1)

    def apply(self, local_rays: RayBundle) -> RayBundle:
        length = self.length
        o, d = local_rays.origins, local_rays.directions
        # chain (entrance z=0, light -z) -> CPC (entrance z=length, exit z=0):
        o_cpc = jnp.stack([o[:, 0], o[:, 1], jnp.full(o.shape[0], length)], axis=-1)
        d_cpc = jnp.stack([d[:, 0], d[:, 1], d[:, 2]], axis=-1)

        s, c = self._wall_tilt()
        oe, de, factor, path_add = trace(
            o_cpc, d_cpc, self._wall_normals(), self.exit_apothem,
            s, c, length, self.reflectivity, self.max_bounces,
        )

        o_out = jnp.stack([oe[:, 0], oe[:, 1], jnp.full(oe.shape[0], -length)], axis=-1)
        d_out = jnp.stack([de[:, 0], de[:, 1], de[:, 2]], axis=-1)

        return RayBundle(
            origins=o_out,
            directions=d_out,
            values=local_rays.values * factor,
            path_length=local_rays.path_length + self.index * path_add,
            n=local_rays.n,
        )

    def cross_sections(self) -> tuple[Array, Array]:
        n_slices = 24
        # entrance at z=0 (apothem a1) down to the exit at z=-length (apothem a2)
        z_chain = jnp.linspace(0.0, -self.length, n_slices)
        s, c = self._wall_tilt()
        apothem = profile_apothem(self.length + z_chain, self.exit_apothem, s, c)
        corner_r = apothem / jnp.cos(jnp.pi / self.n_sides)
        ang = (self.orientation + jnp.pi / self.n_sides
               + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides)
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)        # (N, 2)
        rings = corner_r[:, None, None] * unit[None, :, :]            # (K, N, 2)
        return z_chain, rings
