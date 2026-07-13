from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ...core.interactions import reflect
from ...core.ray_bundle import RayBundle
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

    A new cone type therefore only describes its meridian profile
    (:meth:`_meridian`) and its per-facet intersection (:meth:`_nearest_hit`).
    Every field is static, so the whole cone is a leaf-free pytree the tracer
    can broadcast through ``vmap`` at no runtime cost.
    """

    _N_SLICES = 24  # meridian samples for cross_sections (plain class constant)

    n_sides: eqx.AbstractVar[int]
    orientation: eqx.AbstractVar[float]  # rotation about the optical axis, radians
    entrance_apothem: eqx.AbstractVar[float]  # mouth inradius a1 (at z = length)
    exit_apothem: eqx.AbstractVar[float]  # exit inradius a2 (at z = 0)
    reflectivity: eqx.AbstractVar[float]  # per-bounce wall reflectivity
    max_bounces: eqx.AbstractVar[int]  # reflections traced before absorption

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
    ) -> tuple[Array, Array, Array]:
        """Reflect one ray off a wall hit at parameter ``t``.

        Returns ``(new_origin, new_direction, path_added)``. The new origin is
        nudged just off the wall along the reflected ray so the next intersection
        test sees this wall behind it; the nudge lies on the outgoing ray, so it
        is added back to the optical path and the geometry stays exact.
        """
        refl_d, _ = reflect(d, normal)
        nudge = _NUDGE * self.exit_apothem
        return o + t * d + nudge * refl_d, refl_d, t + nudge

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
        trajectory: ``(steps + 1, N, 3)`` per-step positions (pixel-local
            frame) when the trace was run with ``record_trajectory=True``,
            else ``None``. Diagnostics only -- it costs
            ``(max_bounces + 2) x N x 3`` floats, so it is off by default.
    """

    rays: RayBundle
    bounces: jax.Array
    trajectory: jax.Array | None


def _trace_step(o, d, value, path, done, nmed, hit_stop, bounces, walls, stop, vertex):
    """One trace event for one ray (frozen once ``done``)."""
    t_wall, wall_nrm = walls.nearest_hit(o, d)

    t_stop, p_stop, within = stop._hit(o, d, vertex)
    t_stop = jnp.where(jnp.isfinite(t_stop) & within & (t_stop > 0.0), t_stop, jnp.inf)

    dz = d[2]
    t_ent = jnp.where(dz > 0.0, (walls.length - o[2]) / jnp.where(dz > 0.0, dz, 1.0), jnp.inf)

    is_stop = jnp.isfinite(t_stop) & (t_stop <= t_wall) & (t_stop <= t_ent)
    is_wall = jnp.isfinite(t_wall) & (t_wall < t_stop) & (t_wall <= t_ent)
    is_lost = ~is_stop & ~is_wall

    tw = jnp.where(jnp.isfinite(t_wall), t_wall, 0.0)
    o_wall, refl_d, seg_wall = walls.reflect_ray(o, d, tw, wall_nrm)

    # Terminate on the surface: keep the throughput and the true incident direction
    ts = jnp.where(jnp.isfinite(t_stop), t_stop, 0.0)

    o_new = jnp.where(is_wall, o_wall, jnp.where(is_stop, p_stop, o))
    d_new = jnp.where(is_wall, refl_d, d)
    value_new = jnp.where(is_lost, 0.0, jnp.where(is_wall, value * walls.reflectivity, value))
    path_new = path + jnp.where(is_wall, seg_wall, jnp.where(is_stop, ts * nmed, 0.0))
    done_new = done | is_stop | is_lost

    live = ~done
    return (
        jnp.where(live, o_new, o),
        jnp.where(live, d_new, d),
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
    absorbed). Rays whose transverse position lies outside the mouth polygon
    strike the dead face around the cone and are lost immediately. Landing on
    ``stop`` is the only surviving outcome: the returned ``rays.alive`` is the
    incoming liveness AND-ed with "landed". Pass ``record_trajectory=True`` to
    also collect the per-step ray positions (diagnostics; memory scales with
    ``max_bounces x n_rays``).
    """
    if max_bounces is None:
        max_bounces = walls.max_bounces
    # pixel-local (mouth z=0) -> cone frame (exit z=0, mouth z=length)
    lift = jnp.array([0.0, 0.0, walls.length])
    vertex = stop.vertex_z + walls.length
    n = rays.origins.shape[0]
    inside = walls.in_mouth(rays.origins[:, :2])
    carry = (
        rays.origins + lift,
        rays.directions,
        jnp.where(inside, rays.values, 0.0),
        rays.path_length,
        ~inside,
        rays.n,
        jnp.zeros(n, bool),
        jnp.zeros(n, jnp.int32),
    )

    step_fn = jax.vmap(_trace_step, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None))

    def step(carry: _Carry, _):
        out = step_fn(*carry, walls, stop, vertex)
        # Emit per-step positions only when the trajectory is recorded.
        return out, out[0] if record_trajectory else None

    carry, traj = jax.lax.scan(step, carry, None, length=max_bounces + 1)
    o, d, value, path, done, nmed, hit_stop, bounces = carry

    value = jnp.where(done, value, 0.0)  # rays still live at the end are absorbed
    landed = hit_stop & done
    trajectory = (
        jnp.concatenate([rays.origins[None], traj - lift], axis=0) if record_trajectory else None
    )
    return ChainTrace(
        rays=RayBundle(
            origins=o - lift,
            directions=d,
            values=value,
            path_length=path,
            n=nmed,
            alive=rays.alive & landed,
        ),
        bounces=bounces,
        trajectory=trajectory,
    )
