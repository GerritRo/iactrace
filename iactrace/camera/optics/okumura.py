from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...core.responses import ResponseCurve
from .polygonal import PolygonalCone
from .winston import cpc_full_length, cpc_ideal_wall_tilt

_T_FLOOR = 1e-6  # spurious-hit rejection floor, scaled by a2
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
    orientation (inward vs. outward) is irrelevant to
    :func:`~iactrace.core.interactions.reflect`.
    """
    Rp = _polyval(gd_r, t)
    Zp = _polyval(gd_z, t)
    nrm = jnp.array([Zp * n[0], Zp * n[1], -Rp])
    norm_sq = nrm @ nrm
    return nrm / jnp.sqrt(jnp.where(norm_sq > 0.0, norm_sq, 1.0))


class OkumuraCone(PolygonalCone):
    """Okumura light collector: a polygonal cone with Bezier-curve walls.

    A hollow light guide whose ``n_sides`` walls follow a quadratic or cubic
    Bezier meridian (Okumura 2012, arXiv:1205.3968) rather than the Winston
    paraboloid. Construct it either from an explicit list of interior control
    points or via :meth:`quadratic` / :meth:`cubic`, using the relative
    coordinates tabulated in the paper. Like the Winston cone, it answers the
    per-facet Bezier-meridian hit (:meth:`_nearest_hit`); the bounce loop is
    owned by the shared :func:`~iactrace.camera.optics.polygonal.trace_chain`.

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
        reflectivity_curve: Optional coating curve
            (:class:`~iactrace.core.responses.ResponseCurve`) multiplying the scalar,
            evaluated at each bounce's actual incidence angle and at the ray's
            wavelength; ``None`` (default) is a flat wall response.
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
    reflectivity_curve: ResponseCurve | None
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
        reflectivity_curve: ResponseCurve | None = None,
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
            s, c = cpc_ideal_wall_tilt(a2, a1)
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
        self.reflectivity_curve = reflectivity_curve
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

    def _nearest_hit(self, o: Array, d: Array) -> tuple[Array, Array]:
        r_c = jnp.asarray(self.r_coeffs)
        z_c = jnp.asarray(self.z_coeffs)
        gd_r = _polyder(r_c)
        gd_z = _polyder(z_c)
        tau_all, t_all = jax.vmap(
            lambda nh: _wall_hit(o, d, nh, r_c, z_c, gd_r, gd_z, self.exit_apothem)
        )(self.n_hats)
        kbest = jnp.argmin(tau_all)
        normal = _wall_normal(self.n_hats[kbest], t_all[kbest], gd_r, gd_z)
        return tau_all[kbest], normal

    def _meridian(self) -> tuple[Array, Array]:
        t = jnp.linspace(1.0, 0.0, self._N_SLICES)
        apothem = _polyval(jnp.asarray(self.r_coeffs), t)
        z_chain = _polyval(jnp.asarray(self.z_coeffs), t) - self.length
        return z_chain, apothem