from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.interactions import reflect
from ..core.ray_bundle import RayBundle
from .concentrator import Concentrator
from .winston_cone import cpc_full_length

_NUDGE = 1e-5  # off-wall step, scaled by a2
_T_FLOOR = 1e-6  # spurious-hit rejection floor
_N_BRACKET = 12  # sub-intervals used to isolate meridian roots on t in [0, 1]
_N_BISECT = 20  # bisection steps that shrink each bracket
_N_POLISH = 2  # Newton steps that give the selected root a clean derivative


def _bezier_power_coeffs(control_values: Sequence[float]) -> list[float]:
    """Power-basis coefficients ``[a0, ..., ad]`` of a 1-D Bezier curve.

    Converts Bernstein control values ``c0..cd`` (the curve passes through
    ``c0`` at ``t = 0`` and ``cd`` at ``t = 1``) to ``B(t) = sum_k a_k t**k``
    via the standard identity
    ``a_k = C(d, k) * sum_{j<=k} (-1)^{k-j} C(k, j) c_j``.
    """
    c = [float(v) for v in control_values]
    d = len(c) - 1
    coeffs: list[float] = []
    for k in range(d + 1):
        s = 0.0
        for j in range(k + 1):
            s += (-1.0) ** (k - j) * math.comb(k, j) * c[j]
        coeffs.append(math.comb(d, k) * s)
    return coeffs


def _polyval(coeffs: Array, t: Array) -> Array:
    """Horner evaluation of a polynomial with ``coeffs`` ordered low -> high."""
    res = coeffs[-1] * jnp.ones_like(t)
    for k in range(coeffs.shape[0] - 2, -1, -1):
        res = res * t + coeffs[k]
    return res


def _polyder(coeffs: Array) -> Array:
    """Power-basis coefficients of the derivative (low -> high)."""
    if coeffs.shape[0] == 1:
        return jnp.zeros_like(coeffs)
    ks = jnp.arange(1, coeffs.shape[0], dtype=coeffs.dtype)
    return coeffs[1:] * ks


def _meridian_coeffs(
    control_points: Sequence[tuple[float, float]],
    entrance_apothem: float,
    exit_apothem: float,
    length: float,
) -> tuple[list[float], list[float]]:
    """Power-basis coefficients of the meridian ``R(t)`` and ``Z(t)``.

    ``control_points`` are the *interior* Bezier points in the paper's
    normalized box; the exit rim ``(0, 0)`` and mouth ``(1, 1)`` are implied.
    """
    r_vals = [0.0, *(p[0] for p in control_points), 1.0]
    z_vals = [0.0, *(p[1] for p in control_points), 1.0]
    br = _bezier_power_coeffs(r_vals)
    bz = _bezier_power_coeffs(z_vals)
    dr = entrance_apothem - exit_apothem
    r_coeffs = [dr * b for b in br]
    r_coeffs[0] += exit_apothem
    z_coeffs = [length * b for b in bz]
    return r_coeffs, z_coeffs


def _wall_hit(
    o: Array,
    d: Array,
    n: Array,
    r_coeffs: Array,
    z_coeffs: Array,
    gd_r: Array,
    gd_z: Array,
    a2: float,
) -> tuple[Array, Array]:
    """Smallest forward ray parameter hitting facet ``n`` and its Bezier ``t``.

    Returns ``(tau, t)``; ``tau = inf`` when the facet is not hit in front of
    the ray. ``gd_r`` / ``gd_z`` are the derivative coefficients of ``R`` / ``Z``.
    """
    p = d[0] * n[0] + d[1] * n[1]
    q = d[2]
    u0 = o[0] * n[0] + o[1] * n[1]
    z0 = o[2]

    # G(t) = q R(t) - p Z(t) + (p z0 - q u0), and its derivative coefficients.
    g = q * r_coeffs - p * z_coeffs
    g = g.at[0].add(p * z0 - q * u0)
    gd = q * gd_r - p * gd_z

    # Isolate roots on [0, 1] by sign changes over a fixed set of sub-intervals.
    ts = jnp.linspace(0.0, 1.0, _N_BRACKET + 1)
    gs = _polyval(g, ts)
    lo, hi = ts[:-1], ts[1:]
    glo, ghi = gs[:-1], gs[1:]
    bracketed = (glo * ghi) < 0.0

    def _bisect(carry, _):
        lo, hi, glo = carry
        mid = 0.5 * (lo + hi)
        gmid = _polyval(g, mid)
        left = (glo * gmid) <= 0.0
        return (
            jnp.where(left, lo, mid),
            jnp.where(left, mid, hi),
            jnp.where(left, glo, gmid),
        ), None

    (lo, hi, _), _ = jax.lax.scan(_bisect, (lo, hi, glo), None, length=_N_BISECT)
    t = 0.5 * (lo + hi)

    # Newton polish: gives the selected root a clean, correct derivative
    def _polish(t, _):
        gt = _polyval(g, t)
        gp = _polyval(gd, t)
        safe = jnp.where(jnp.abs(gp) > 1e-30, gp, 1.0)
        step = jnp.where(jnp.abs(gp) > 1e-30, gt / safe, 0.0)
        return jnp.clip(t - step, lo, hi), None

    t, _ = jax.lax.scan(_polish, t, None, length=_N_POLISH)

    # Recover the ray parameter from whichever coordinate the ray moves along
    R = _polyval(r_coeffs, t)
    Z = _polyval(z_coeffs, t)
    use_p = jnp.abs(p) >= jnp.abs(q)
    denom = jnp.where(use_p, p, q)
    safe = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    tau = jnp.where(use_p, R - u0, Z - z0) / safe

    ok = (
        bracketed
        & (t >= 0.0)
        & (t <= 1.0)
        & jnp.isfinite(tau)
        & (tau > _T_FLOOR * a2)
        & (jnp.maximum(jnp.abs(p), jnp.abs(q)) > 1e-30)
    )
    tau = jnp.where(ok, tau, jnp.inf)
    k = jnp.argmin(tau)
    return tau[k], t[k]


def _wall_normal(n: Array, t: Array, gd_r: Array, gd_z: Array) -> Array:
    """Unit wall normal on facet ``n`` at Bezier parameter ``t``.

    The meridian tangent in the ``(u, z)`` half-plane is ``(R'(t), Z'(t))``; the
    surface normal is perpendicular to it, lifted to 3-D along ``n_hat``. The
    orientation (inward vs. outward) is irrelevant to :func:`reflect`.
    """
    Rp = _polyval(gd_r, t)
    Zp = _polyval(gd_z, t)
    nrm = jnp.array([Zp * n[0], Zp * n[1], -Rp])
    norm_sq = nrm @ nrm
    return nrm / jnp.sqrt(jnp.where(norm_sq > 0.0, norm_sq, 1.0))


def _single_step(o, d, value, path, done, n_hats, r_c, z_c, gd_r, gd_z, a2, length, refl):
    """One reflection event for one ray (frozen once ``done``)."""
    tau_all, t_all = jax.vmap(lambda n: _wall_hit(o, d, n, r_c, z_c, gd_r, gd_z, a2))(n_hats)
    kbest = jnp.argmin(tau_all)
    t_wall = tau_all[kbest]
    t_bez = t_all[kbest]

    dz = d[2]
    safe_dz = jnp.where(dz != 0.0, dz, 1.0)
    t_exit = jnp.where(dz < 0, (0.0 - o[2]) / safe_dz, jnp.inf)
    t_ent = jnp.where(dz > 0, (length - o[2]) / safe_dz, jnp.inf)

    is_exit = jnp.isfinite(t_exit) & (t_exit <= t_wall) & (t_exit <= t_ent)
    is_wall = jnp.isfinite(t_wall) & (t_wall < t_exit) & (t_wall <= t_ent)
    is_lost = (~is_exit) & (~is_wall)

    tw = jnp.where(jnp.isfinite(t_wall), t_wall, 0.0)
    pw = o + tw * d
    nrm = _wall_normal(n_hats[kbest], t_bez, gd_r, gd_z)
    dw, _ = reflect(d, nrm)
    # Step off the wall along the reflected ray so the next intersection test
    # sees this wall behind it. The nudge lies on the outgoing ray, so it is
    # added back to the optical path and the geometry stays exact.
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


def trace(
    origins,
    directions,
    n_hats,
    r_coeffs,
    z_coeffs,
    entrance_apothem,
    exit_apothem,
    length,
    reflectivity,
    max_bounces,
):
    """Trace rays through the Okumura cone in CPC coords (exit z=0, mouth z=length).

    Returns ``(exit_origins, exit_directions, value_factor, path_added)``;
    ``value_factor`` is ``reflectivity**bounces`` for transmitted rays and ``0``
    for rays that miss the mouth, leave back through it, or never exit within
    ``max_bounces``.
    """
    a1, a2 = entrance_apothem, exit_apothem
    r_c = jnp.asarray(r_coeffs)
    z_c = jnp.asarray(z_coeffs)
    gd_r = _polyder(r_c)
    gd_z = _polyder(z_c)
    n = origins.shape[0]

    # Rays entering outside the cone mouth (inradius a1 at z = length) are lost.
    u_all = origins[:, :2] @ n_hats.T
    inside = jnp.all(u_all <= a1 + 1e-9, axis=1)

    carry = (
        origins,
        directions,
        jnp.where(inside, 1.0, 0.0),
        jnp.zeros(n),
        ~inside,
    )

    step_fn: Callable[..., tuple[Array, Array, Array, Array, Array]] = jax.vmap(
        _single_step,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
    )

    def step(carry, _):
        out = step_fn(*carry, n_hats, r_c, z_c, gd_r, gd_z, a2, length, reflectivity)
        return out, None

    (o, d, value, path, done), _ = jax.lax.scan(
        step,
        carry,
        None,
        length=max_bounces + 1,
    )
    # Rays still bouncing at the end are absorbed.
    value = jnp.where(done, value, 0.0)
    return o, d, value, path


# OkumuraCone concentrator


class OkumuraCone(Concentrator):
    """Okumura light collector: a polygonal cone with Bezier-curve walls.

    A hollow light guide whose ``n_sides`` walls follow a quadratic or cubic
    Bezier meridian (Okumura 2012, arXiv:1205.3968) rather than the Winston
    paraboloid. Construct it either from an explicit list of interior control
    points or via :meth:`quadratic` / :meth:`cubic`, using the relative
    coordinates tabulated in the paper.

    The control points are given in the paper's normalized box: the exit rim is
    ``(0, 0)`` and the mouth is ``(1, 1)``, so a control point ``(r, z)`` has
    ``r`` interpolating the inradius from ``exit_apothem`` to ``entrance_apothem``
    and ``z`` interpolating the axial position from the exit plane to the mouth.

    Args:
        n_sides: Number of facets (6 = hexagonal, 4 = square, ...).
        entrance_apothem: Mouth inradius ``a1`` (the apothem at ``z = length``).
        exit_apothem: Exit aperture inradius ``a2``.
        control_points: Interior Bezier control points in normalized
            coordinates -- ``[(P1r, P1z)]`` for a quadratic curve,
            ``[(P1r, P1z), (P2r, P2z)]`` for a cubic one. The endpoints
            ``(0, 0)`` (exit) and ``(1, 1)`` (mouth) are implied.
        length: Physical depth. ``None`` defaults to the length of the
            equivalent full Winston cone, ``L = (a1 + a2) * cos/sin(theta_max)``
            with ``sin(theta_max) = a2 / a1`` -- the same ``L`` Okumura compares
            against, so the Okumura cone is a true drop-in for that Winston cone.
        reflectivity: Per-bounce wall reflectivity (scalar).
        max_bounces: Maximum reflections traced before a ray is absorbed.
        orientation_deg: Rotation of the polygon about the optical axis.

    Raises:
        ValueError: if ``0 < exit_apothem < entrance_apothem`` is violated, if
            no interior control point is given, or if the control points give a
            non-monotonic axial profile ``Z(t)`` (an ill-defined depth).
    """

    n_sides: int = eqx.field(static=True)
    exit_apothem: float = eqx.field(static=True)
    entrance_apothem: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    control_points: tuple[tuple[float, float], ...] = eqx.field(static=True)
    r_coeffs: tuple[float, ...] = eqx.field(static=True)
    z_coeffs: tuple[float, ...] = eqx.field(static=True)
    reflectivity: float = eqx.field(static=True)
    max_bounces: int = eqx.field(static=True)
    orientation: float = eqx.field(static=True)  # radians

    def __init__(
        self,
        n_sides: int,
        entrance_apothem: float,
        exit_apothem: float,
        control_points: Sequence[tuple[float, float]],
        length: float | None = None,
        reflectivity: float = 0.9,
        max_bounces: int = 10,
        orientation_deg: float = 0.0,
    ) -> None:
        a1, a2 = float(entrance_apothem), float(exit_apothem)
        if not 0.0 < a2 < a1:
            raise ValueError(
                "require 0 < exit_apothem < entrance_apothem, got "
                f"exit_apothem={a2}, entrance_apothem={a1}"
            )
        pts = tuple((float(r), float(z)) for r, z in control_points)
        if len(pts) < 1:
            raise ValueError(
                "control_points must define at least one interior Bezier point "
                "(one point -> quadratic curve, two -> cubic)."
            )

        if length is None:
            s = a2 / a1
            c = math.sqrt(1.0 - s * s)
            length = cpc_full_length(a2, s, c)
        length = float(length)
        if length <= 0.0:
            raise ValueError(f"length must be > 0, got {length}")

        r_coeffs, z_coeffs = _meridian_coeffs(pts, a1, a2, length)

        # A well-defined cone needs a monotonic axial profile so that t in [0, 1]
        # maps one-to-one onto z in [0, length].
        tt = np.linspace(0.0, 1.0, 257)
        zz = np.polyval(list(reversed(z_coeffs)), tt)
        if np.any(np.diff(zz) < -1e-9 * length):
            raise ValueError(
                "control_points yield a non-monotonic axial profile Z(t); the "
                "cone depth is ill-defined. Keep the z-coordinate of each "
                "control point increasing (0 < P1z < ... < 1)."
            )

        self.n_sides = int(n_sides)
        self.exit_apothem = a2
        self.entrance_apothem = a1
        self.length = length
        self.control_points = pts
        self.r_coeffs = tuple(r_coeffs)
        self.z_coeffs = tuple(z_coeffs)
        self.reflectivity = float(reflectivity)
        self.max_bounces = int(max_bounces)
        self.orientation = math.radians(float(orientation_deg))

    @classmethod
    def quadratic(
        cls,
        n_sides: int,
        entrance_apothem: float,
        exit_apothem: float,
        p1: tuple[float, float],
        **kwargs,
    ) -> OkumuraCone:
        """Build a quadratic Okumura cone from its single Bezier control point ``P1``."""
        return cls(n_sides, entrance_apothem, exit_apothem, [p1], **kwargs)

    @classmethod
    def cubic(
        cls,
        n_sides: int,
        entrance_apothem: float,
        exit_apothem: float,
        p1: tuple[float, float],
        p2: tuple[float, float],
        **kwargs,
    ) -> OkumuraCone:
        """Build a cubic Okumura cone from its Bezier control points ``P1`` and ``P2``."""
        return cls(n_sides, entrance_apothem, exit_apothem, [p1, p2], **kwargs)

    @property
    def degree(self) -> int:
        """Degree of the Bezier meridian (2 = quadratic, 3 = cubic)."""
        return len(self.control_points) + 1

    def _wall_normals(self) -> Array:
        a = self.orientation + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides
        return jnp.stack([jnp.cos(a), jnp.sin(a)], axis=-1)

    def apply(self, local_rays: RayBundle) -> RayBundle:
        length = self.length
        o, d = local_rays.origins, local_rays.directions
        # chain (entrance z=0, light -z) -> CPC (entrance z=length, exit z=0):
        o_cpc = jnp.stack([o[:, 0], o[:, 1], jnp.full(o.shape[0], length)], axis=-1)

        oe, de, factor, path_add = trace(
            o_cpc,
            d,
            self._wall_normals(),
            jnp.asarray(self.r_coeffs),
            jnp.asarray(self.z_coeffs),
            self.entrance_apothem,
            self.exit_apothem,
            length,
            self.reflectivity,
            self.max_bounces,
        )

        o_out = jnp.stack([oe[:, 0], oe[:, 1], jnp.full(oe.shape[0], -length)], axis=-1)

        return RayBundle(
            origins=o_out,
            directions=de,
            values=local_rays.values * factor,
            path_length=local_rays.path_length + self.index * path_add,
            n=local_rays.n,
        )

    def cross_sections(self) -> tuple[Array, Array]:
        n_slices = 24
        r_c = jnp.asarray(self.r_coeffs)
        z_c = jnp.asarray(self.z_coeffs)
        # sample the meridian from the mouth (t=1, z=0 chain) to the exit
        # (t=0, z=-length chain)
        t = jnp.linspace(1.0, 0.0, n_slices)
        apothem = _polyval(r_c, t)
        z_chain = _polyval(z_c, t) - self.length
        corner_r = apothem / jnp.cos(jnp.pi / self.n_sides)
        ang = (
            self.orientation
            + jnp.pi / self.n_sides
            + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides
        )
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # (N, 2)
        rings = corner_r[:, None, None] * unit[None, :, :]  # (K, N, 2)
        return z_chain, rings
