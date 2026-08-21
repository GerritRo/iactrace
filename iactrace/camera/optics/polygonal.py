from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ...core.interactions import reflect
from ...core.ray_bundle import RayBundle
from ...core.responses import ResponseCurve
from ...core.trajectory import TraceResult, Trajectory
from ..detector.surface import DetectionSurface
from .concentrator import Concentrator

_NUDGE = 1e-5  # off-wall step, scaled by the exit apothem

# (origins, directions, values, path_length, done, n, hit_stop, bounces)
_Carry = tuple[Array, Array, Array, Array, Array, Array, Array, Array]


class PolygonalCone(Concentrator):
    """A hollow reflective cone whose ``n_sides`` facets are lofted around a
    meridian: one concrete :class:`~iactrace.camera.optics.concentrator.Concentrator`
    that delivers rays by bouncing them off its reflecting walls.

    * **Geometry.** The facet plane normals (:attr:`n_hats`) and the drawable
      cross-sections depend only on the polygon (``n_sides``, ``orientation``)
      and on the meridian samples the subclass provides via :meth:`_meridian`.
    * **Wall tracing.** :meth:`to_surface` runs the shared :func:`trace_chain`
      bounce loop, which asks the cone for the raw nearest wall hit
      (:meth:`_nearest_hit`); the cavity clamp (:meth:`nearest_hit`), the
      mouth-aperture mask (:meth:`in_mouth`) and the off-wall reflection
      book-keeping (:meth:`reflect_ray`) are shared here. All tracer coordinates
      are the cone frame (exit ``z = 0``, mouth ``z = length``).
    * **Wall response.** Every bounce costs :meth:`wall_reflectivity`, evaluated
      at the incidence angle of *that* bounce and at the ray's wavelength.
    """

    _N_SLICES = 24  # meridian samples for cross_sections (plain class constant)

    n_sides: eqx.AbstractVar[int]
    orientation: eqx.AbstractVar[float]  # rotation about the optical axis, radians
    entrance_apothem: eqx.AbstractVar[float]  # mouth inradius a1 (at z = length)
    exit_apothem: eqx.AbstractVar[float]  # exit inradius a2 (at z = 0)
    reflectivity: eqx.AbstractVar[float]  # per-bounce wall reflectivity (bulk scalar)
    reflectivity_curve: eqx.AbstractVar[ResponseCurve | None]  # optional R(angle, wavelength)
    max_bounces: eqx.AbstractVar[int]  # reflections traced before absorption

    def wall_reflectivity(self, cos_theta_i: Array, wavelength: Array) -> Array:
        """Per-ray wall reflectivity, shape ``(n_rays,)``.

        The scalar :attr:`reflectivity` is the flat bulk value; an optional
        :attr:`reflectivity_curve` multiplies in the coating response,
        evaluated at each ray's *actual* incidence angle on the wall and at its
        wavelength.

        Args:
            cos_theta_i: Cosine of the angle between the ray and the wall
                normal at the hit point, shape ``(n_rays,)``. ``1`` is normal
                incidence, ``0`` grazing.
            wavelength: Per-ray wavelength, shape ``(n_rays,)``.
        """
        if self.reflectivity_curve is None:
            return jnp.full(cos_theta_i.shape, self.reflectivity)
        idx = jnp.zeros(cos_theta_i.shape, dtype=jnp.int32)  # the cone is one element
        curve = self.reflectivity_curve(cos_theta_i, idx, wavelength)
        return self.reflectivity * curve

    @property
    def n_hats(self) -> Array:
        """(M, 2) inward plane normals of the polygon facets."""
        return self._wall_normals()

    def _wall_normals(self) -> Array:
        a = self.orientation + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides
        return jnp.stack([jnp.cos(a), jnp.sin(a)], axis=-1)

    @abstractmethod
    def _nearest_hit(self, o: Array, d: Array) -> tuple[Array, Array]:
        """Raw nearest forward wall hit for one ray on the extended walls.

        Returns ``(t, normal)`` with ``t = inf`` when no facet is hit in front
        of the ray; ``normal`` may be arbitrary then (it is never consumed).
        The hit is on the *unbounded* meridian surface -- :meth:`nearest_hit`
        clamps it to the cavity.
        """
        raise NotImplementedError

    def nearest_hit(self, o: Array, d: Array) -> tuple[Array, Array]:
        """Nearest forward wall hit for one ray, clamped to ``z in [0, length]``.

        Returns ``(t, normal)`` with ``t = inf`` when the nearest wall root falls
        outside the cavity -- e.g. the ray has dropped below the exit, where the
        infinite wall surface would otherwise give a spurious root.
        """
        t, normal = self._nearest_hit(o, d)
        z_hit = o[2] + t * d[2]
        inside = (z_hit >= -1e-9) & (z_hit <= self.length + 1e-9)
        return jnp.where(jnp.isfinite(t) & inside, t, jnp.inf), normal

    def reflect_ray(
        self, o: Array, d: Array, t: Array, normal: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Reflect one ray off a wall hit at parameter ``t``.

        Returns ``(new_origin, new_direction, path_added, cos_theta_i)``. The new
        origin is nudged just off the wall along the reflected ray so the next
        intersection test sees this wall behind it; the nudge lies on the outgoing
        ray, so it is added back to the optical path and the geometry stays exact.
        """
        refl_d, cos_i = reflect(d, normal)
        nudge = _NUDGE * self.exit_apothem
        cos_theta_i = jnp.clip(jnp.abs(cos_i.squeeze(-1)), 0.0, 1.0)
        return o + t * d + nudge * refl_d, refl_d, t + nudge, cos_theta_i

    def in_mouth(self, xy: Array) -> Array:
        """True where the transverse position ``xy`` lies inside the mouth polygon."""
        return jnp.all(xy @ self.n_hats.T <= self.entrance_apothem + 1e-9, axis=-1)

    def to_surface(self, rays: RayBundle, surface: DetectionSurface) -> RayBundle:
        """Trace *rays* through the reflecting walls onto *surface*.

        The wall-based implementation of
        :meth:`~iactrace.camera.optics.concentrator.Concentrator.to_surface`: the
        shared :func:`trace_chain` bounces rays off the cavity walls and lands
        them on *surface*, co-traced so a sensor surface peeking into the cavity is
        hit mid-bounce.
        """
        return trace_chain(self, surface, rays).rays

    def trace_to_surface(self, rays: RayBundle, surface: DetectionSurface) -> TraceResult:
        """:meth:`to_surface`, also returning the per-bounce wall path."""
        tr = trace_chain(self, surface, rays, record_trajectory=True)
        return TraceResult(tr.rays, tr.trajectory)

    @abstractmethod
    def _meridian(self) -> tuple[Array, Array]:
        """``(z, apothem)`` samples of the wall meridian, ``_N_SLICES`` each.

        In the pixel-local frame, from the entrance (``z[0] = 0``, mouth
        apothem) down to the exit (``z[-1] = -length``, exit apothem).
        """
        raise NotImplementedError

    def cross_sections(self) -> tuple[Array, Array]:
        z_chain, apothem = self._meridian()
        corner_r = apothem / jnp.cos(jnp.pi / self.n_sides)
        ang = (
            self.orientation
            + jnp.pi / self.n_sides
            + 2.0 * jnp.pi * jnp.arange(self.n_sides) / self.n_sides
        )
        unit = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # (M, 2)
        rings = corner_r[:, None, None] * unit[None, :, :]  # (K, M, 2)
        return z_chain, rings


class ChainTrace(NamedTuple):
    """Result of :func:`trace_chain`.

    Attributes:
        rays: Terminated :class:`RayBundle`; ``alive`` marks the rays that
            landed on the surface, ``values`` carry the throughput delivered
            there (0 for lost / absorbed rays), ``origins`` the landing point,
            ``directions`` the true incident direction, ``path_length`` the
            accumulated OPL.
        bounces: ``(N,)`` int -- wall reflections before termination.
        trajectory: A :class:`~iactrace.core.trajectory.Trajectory` of per-step
            positions (pixel-local frame) when the trace was run with
            ``record_trajectory=True``, else ``None``. Diagnostics only -- it
            costs ``(max_bounces + 2) x N x 3`` floats, so it is off by default.
    """

    rays: RayBundle
    bounces: jax.Array
    trajectory: Trajectory | None


class _Event(NamedTuple):
    """The nearest trace event, as seen by :func:`_hit_event`.

    Attributes:
        o_wall, refl_d, seg_wall: reflected ray and path increment, valid where
            ``is_wall``.
        cos_theta_i: incidence cosine of *this* bounce, what the wall coating is
            evaluated at.
        p_stop, t_stop: landing point and ray parameter, valid where ``is_stop``.
        is_stop, is_wall: which of the two happened; neither means the ray left
            through the entrance (lost).
    """

    o_wall: Array
    refl_d: Array
    seg_wall: Array
    cos_theta_i: Array
    p_stop: Array
    t_stop: Array
    is_stop: Array
    is_wall: Array


def _hit_event(o, d, walls, stop, vertex) -> _Event:
    """Nearest event for one ray: a wall bounce, a landing on ``stop``, or a loss.

    Pure geometry -- no throughput book-keeping -- so the per-bounce wall
    reflectivity can be evaluated on the whole bundle at once (a
    :class:`~iactrace.core.responses.ResponseCurve` is a batched callable) before
    :func:`_advance` folds it in.
    """
    t_wall, wall_nrm = walls.nearest_hit(o, d)

    t_stop, p_stop, within = stop._hit(o, d, vertex)
    t_stop = jnp.where(jnp.isfinite(t_stop) & within & (t_stop > 0.0), t_stop, jnp.inf)

    dz = d[2]
    t_ent = jnp.where(dz > 0.0, (walls.length - o[2]) / jnp.where(dz > 0.0, dz, 1.0), jnp.inf)

    is_stop = jnp.isfinite(t_stop) & (t_stop <= t_wall) & (t_stop <= t_ent)
    is_wall = jnp.isfinite(t_wall) & (t_wall < t_stop) & (t_wall <= t_ent)

    tw = jnp.where(jnp.isfinite(t_wall), t_wall, 0.0)
    o_wall, refl_d, seg_wall, cos_theta_i = walls.reflect_ray(o, d, tw, wall_nrm)

    # Terminate on the surface: keep the throughput and the true incident direction
    ts = jnp.where(jnp.isfinite(t_stop), t_stop, 0.0)
    return _Event(o_wall, refl_d, seg_wall, cos_theta_i, p_stop, ts, is_stop, is_wall)


def _advance(carry: _Carry, ev: _Event, refl: Array) -> _Carry:
    """Apply one trace event to the whole bundle (rays frozen once ``done``).

    ``refl`` is the per-ray wall reflectivity of *this* bounce, already
    evaluated at each ray's incidence angle and wavelength.
    """
    o, d, value, path, done, nmed, hit_stop, bounces = carry
    is_wall, is_stop = ev.is_wall, ev.is_stop
    is_lost = ~is_stop & ~is_wall

    vec_wall, vec_stop = is_wall[:, None], is_stop[:, None]
    o_new = jnp.where(vec_wall, ev.o_wall, jnp.where(vec_stop, ev.p_stop, o))
    d_new = jnp.where(vec_wall, ev.refl_d, d)
    value_new = jnp.where(is_lost, 0.0, jnp.where(is_wall, value * refl, value))
    path_new = path + jnp.where(is_wall, ev.seg_wall, jnp.where(is_stop, ev.t_stop * nmed, 0.0))
    done_new = done | is_stop | is_lost

    live = ~done
    return (
        jnp.where(live[:, None], o_new, o),
        jnp.where(live[:, None], d_new, d),
        jnp.where(live, value_new, value),
        jnp.where(live, path_new, path),
        jnp.where(live, done_new, done),
        nmed,
        hit_stop | (live & is_stop),
        bounces + (live & is_wall).astype(bounces.dtype),
    )


def trace_chain(
    walls: PolygonalCone,
    stop: DetectionSurface,
    rays: RayBundle,
    max_bounces: int | None = None,
    record_trajectory: bool = False,
) -> ChainTrace:
    """Trace *rays* through a cone's walls onto the stopping surface ``stop``.

    ``walls`` is the wall-based concentrator itself (a :class:`PolygonalCone`):
    it answers the raw nearest wall hit (:meth:`PolygonalCone._nearest_hit`) and
    the reflection book-keeping, while this function owns the shared bounce loop.

    Everything is the pixel-local frame: *rays* enter at the mouth plane
    ``z = 0`` travelling toward ``-z``, the walls span ``z in [-length, 0]``,
    and ``stop`` sits at its ``vertex_z`` -- below the cone exit (a gap) or
    protruding into the cavity, with no special-casing. (Internally the loop
    runs in the cone frame, exit at ``z = 0``; the shift is invisible from
    outside.)

    Each step every live ray takes the nearest of a wall reflection, a landing
    on ``stop``, or a back-loss through the entrance, until it terminates or
    ``max_bounces`` (default ``walls.max_bounces``) is exhausted (then
    absorbed). Pass ``record_trajectory=True`` to also collect the per-step
     ray positions as a :class:`~iactrace.core.trajectory.Trajectory` on
    :attr:`ChainTrace.trajectory` (diagnostics; memory scales with
    ``max_bounces x n_rays``).
    """
    if max_bounces is None:
        max_bounces = walls.max_bounces
    # pixel-local (mouth z=0) -> cone frame (exit z=0, mouth z=length)
    lift = jnp.array([0.0, 0.0, walls.length])
    vertex = stop.vertex_z + walls.length
    n = rays.origins.shape[0]
    inside = walls.in_mouth(rays.origins[:, :2])

    # Only rays that are inside and alive propagate
    entered = inside & rays.alive
    carry = (
        rays.origins + lift,
        rays.directions,
        jnp.where(inside, rays.values, 0.0),
        rays.path_length,
        ~entered,
        rays.n,
        jnp.zeros(n, bool),
        jnp.zeros(n, jnp.int32),
    )

    event_fn = jax.vmap(_hit_event, in_axes=(0, 0, None, None, None))

    def step(carry: _Carry, _):
        ev = event_fn(carry[0], carry[1], walls, stop, vertex)
        refl = walls.wall_reflectivity(ev.cos_theta_i, rays.wavelength)
        out = _advance(carry, ev, refl)
        # Emit per-step positions only when the trajectory is recorded.
        return out, out[0] if record_trajectory else None

    carry, traj = jax.lax.scan(step, carry, None, length=max_bounces + 1)
    o, d, value, path, done, nmed, hit_stop, bounces = carry

    value = jnp.where(done, value, 0.0)  # rays still live at the end are absorbed
    landed = hit_stop & done
    trajectory = (
        Trajectory(points=jnp.concatenate([rays.origins[None], traj - lift], axis=0))
        if record_trajectory
        else None
    )
    return ChainTrace(
        rays=RayBundle(
            origins=o - lift,
            directions=d,
            values=value,
            path_length=path,
            n=nmed,
            wavelength=rays.wavelength,
            alive=rays.alive & landed,
        ),
        bounces=bounces,
        trajectory=trajectory,
    )
