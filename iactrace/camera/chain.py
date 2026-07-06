from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..core.ray_bundle import RayBundle
from .concentrator import Concentrator
from .photosensor import PhotoSensor, StopSurface
from .winston_cone import ConeWalls


class DetectionChain(eqx.Module):
    """A pixel's detection train: (optional concentrator) -> surface -> photosensor.

    Every chain traces rays up to the photosensor's own stopping surface (its
    photocathode geometry, :meth:`~iactrace.camera.photosensor.PhotoSensor.stopping_surface`)
    and hands the resulting bundle back to the photosensor, which applies its
    detection efficiencies (QE, window response, ...). Geometry is owned by the
    photosensor; the chain only *places* it, at the detector plane
    :attr:`detector_z` set by the concentrator + ``gap``. The chain is identical
    for every pixel in a SensorGroup, so it runs once over all rays at once.

    Attributes:
        concentrator: Optional light concentrator (cone / lightguide).
        photosensor: Photosensor -- both the response and (via its
            ``stopping_surface``) the photocathode geometry rays are traced to.
        gap: Spacing from the concentrator exit (or the entrance with no cone) to
            the detector plane where the photocathode is mounted. Defaults ``0.0``.
    """

    concentrator: Concentrator | None
    photosensor: PhotoSensor
    gap: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if not self.gap >= 0.0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")

    @property
    def detector_z(self) -> float:
        """Detector-plane position in the pixel-local frame: ``-(length + gap)``."""
        length = self.concentrator.length if self.concentrator is not None else 0.0
        return -float(length) - self.gap

    def _effective_surface(self) -> StopSurface:
        """The photosensor's stopping surface, placed at :attr:`detector_z`.

        The photosensor owns the surface with ``vertex_z`` relative to the
        detector plane; here it is shifted into absolute pixel-local coordinates.
        A photosensor without a surface gets a flat plane at the detector plane.
        """
        surf = self.photosensor.stopping_surface()
        if surf is None:
            return StopSurface(vertex_z=self.detector_z)
        return surf.shifted(self.detector_z)

    def propagate(self, local_rays: RayBundle) -> RayBundle:
        """Trace *local_rays* to the stopping surface, then hand off to the photosensor.

        ``local_rays`` are in the pixel-local frame (entrance at ``z = 0``). They
        are delivered to :meth:`_effective_surface` -- jointly with the
        concentrator walls when the surface can be hit mid-bounce (a curved or
        protruding photocathode in a Winston cone), otherwise funnelled through
        the concentrator and advanced straight onto the surface. The photosensor
        then weights the handed-over bundle -- rays at the surface, in the
        surface-local frame -- by its own detection efficiency. Optical path
        length is accumulated up to the surface (concentrator fill index on its
        internal leg, ray medium ``n`` on the free legs).
        """
        at_surface, normals = self._to_surface(local_rays, self._effective_surface())
        return self.photosensor.detect(at_surface, normals)

    def _to_surface(self, rays: RayBundle, surface: StopSurface) -> tuple[RayBundle, jax.Array]:
        """Deliver *rays* to *surface*; return ``(rays_at_surface, normals)``."""
        walls = self.concentrator.walls() if self.concentrator is not None else None
        if walls is not None and surface.needs_joint_trace(walls.length):
            return self._joint_trace(rays, walls, surface)
        if self.concentrator is not None:
            rays = self.concentrator.apply(rays)
        return surface.stop(rays)

    def _joint_trace(
        self, rays: RayBundle, walls: ConeWalls, surface: StopSurface
    ) -> tuple[RayBundle, jax.Array]:
        """Trace the cone walls and the surface together (lift to the cone frame)."""
        o = rays.origins
        cone_rays = RayBundle(
            origins=jnp.stack([o[:, 0], o[:, 1], jnp.full(o.shape[0], walls.length)], axis=-1),
            directions=rays.directions,
            values=rays.values,
            path_length=rays.path_length,
            n=rays.n,
            alive=rays.alive,
        )
        max_bounces = getattr(self.concentrator, "max_bounces", 12)
        tr = trace_chain(walls, surface, cone_rays, max_bounces=max_bounces)
        oo = tr.rays.origins
        at_surface = RayBundle(
            origins=jnp.stack([oo[:, 0], oo[:, 1], oo[:, 2] - walls.length], axis=-1),
            directions=tr.rays.directions,
            values=tr.rays.values,
            path_length=tr.rays.path_length,
            n=tr.rays.n,
            alive=tr.rays.alive,
        )
        return at_surface, tr.normals


# Shared cone-walls + stopping-surface tracer


class ChainTrace(NamedTuple):
    """Result of :func:`trace_chain`.

    Attributes:
        rays: Terminated :class:`RayBundle`; ``values`` carry the throughput
            delivered to the surface (0 for lost / absorbed / undetected rays),
            ``origins`` the landing point, ``directions`` the true incident
            direction, ``path_length`` the accumulated OPL.
        normals: ``(N, 3)`` outward surface normal at each landing (``+z`` where
            not landed), handed to the photosensor for incidence.
        hit_stop: ``(N,)`` bool -- ray terminated on the stopping surface.
        cos_land: ``(N,)`` incidence cosine on the stop (0 where not landed).
        bounces: ``(N,)`` int -- wall reflections before termination.
        trajectory: ``(steps + 1, N, 3)`` per-step positions (cone frame), for
            ray-path diagnostics.
    """

    rays: RayBundle
    normals: jax.Array
    hit_stop: jax.Array
    cos_land: jax.Array
    bounces: jax.Array
    trajectory: jax.Array


def _trace_step(
    o, d, value, path, done, nmed, land_nrm, hit_stop, cos_land, bounces, walls, stop, vertex
):
    """One trace event for one ray (frozen once ``done``)."""
    t_wall, wall_nrm = walls.nearest_hit(o, d)

    t_stop, p_stop, s_nrm, within = stop._hit(o, d, vertex)
    t_stop = jnp.where(jnp.isfinite(t_stop) & within & (t_stop > 0.0), t_stop, jnp.inf)

    dz = d[2]
    t_ent = jnp.where(dz > 0.0, (walls.length - o[2]) / jnp.where(dz > 0.0, dz, 1.0), jnp.inf)

    is_stop = jnp.isfinite(t_stop) & (t_stop <= t_wall) & (t_stop <= t_ent)
    is_wall = jnp.isfinite(t_wall) & (t_wall < t_stop) & (t_wall <= t_ent)
    is_lost = ~is_stop & ~is_wall

    tw = jnp.where(jnp.isfinite(t_wall), t_wall, 0.0)
    o_wall, refl_d, seg_wall = walls.reflect_ray(o, d, tw, wall_nrm)

    # Terminate on the surface: keep the throughput and the true incident direction
    cos_i = jnp.clip(jnp.abs(jnp.dot(d, s_nrm)), 0.0, 1.0)
    ts = jnp.where(jnp.isfinite(t_stop), t_stop, 0.0)

    o_new = jnp.where(is_wall, o_wall, jnp.where(is_stop, p_stop, o))
    d_new = jnp.where(is_wall, refl_d, d)
    value_new = jnp.where(is_lost, 0.0, jnp.where(is_wall, value * walls.reflectivity, value))
    path_new = path + jnp.where(is_wall, seg_wall, jnp.where(is_stop, ts * nmed, 0.0))
    done_new = done | is_stop | is_lost

    live = ~done
    took_stop = live & is_stop
    return (
        jnp.where(live, o_new, o),
        jnp.where(live, d_new, d),
        jnp.where(live, value_new, value),
        jnp.where(live, path_new, path),
        jnp.where(live, done_new, done),
        nmed,
        jnp.where(took_stop[..., None], s_nrm, land_nrm),
        hit_stop | took_stop,
        jnp.where(took_stop, cos_i, cos_land),
        bounces + (live & is_wall).astype(bounces.dtype),
    )


def trace_chain(
    walls: ConeWalls,
    stop: StopSurface,
    rays: RayBundle,
    max_bounces: int = 12,
) -> ChainTrace:
    """Trace *rays* (cone frame) through the walls onto the stopping surface.

    Each step every live ray takes the nearest of a wall reflection, a landing on
    ``stop``, or a back-loss through the entrance, until it terminates or
    ``max_bounces`` is exhausted (then absorbed). ``stop`` is positioned by its
    pixel-local ``vertex_z``, mapped into the cone frame as ``vertex_z + length``
    -- so it may sit below the exit (a gap) or protrude into the cavity, with no
    special-casing.
    """
    vertex = stop.vertex_z + walls.length
    n = rays.origins.shape[0]
    plus_z = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (n, 3))
    carry = (
        rays.origins,
        rays.directions,
        rays.values,
        rays.path_length,
        jnp.zeros(n, bool),
        rays.n,
        plus_z,
        jnp.zeros(n, bool),
        jnp.zeros(n),
        jnp.zeros(n, jnp.int32),
    )

    step_fn = jax.vmap(_trace_step, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None))

    def step(carry, _):
        out = step_fn(*carry, walls, stop, vertex)
        return out, out[0]  # emit positions for the trajectory

    carry, traj = jax.lax.scan(step, carry, None, length=max_bounces + 1)
    o, d, value, path, done, nmed, land_nrm, hit_stop, cos_land, bounces = carry

    value = jnp.where(done, value, 0.0)  # rays still live at the end are absorbed
    landed = hit_stop & done
    trajectory = jnp.concatenate([rays.origins[None], traj], axis=0)
    return ChainTrace(
        rays=RayBundle(
            origins=o,
            directions=d,
            values=value,
            path_length=path,
            n=nmed,
            alive=rays.alive & landed,
        ),
        normals=land_nrm,
        hit_stop=landed,
        cos_land=cos_land,
        bounces=bounces,
        trajectory=trajectory,
    )
