from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .transforms import euler_to_matrix


class RayBundle(eqx.Module):
    """Bundle of rays through the optical system.

    Carries ray positions, directions, weights, path lengths, and a
    per-ray liveness flag.

    The frame of ``origins`` / ``directions`` is implicit and depends on
    where the bundle came from: ``Telescope.render`` and ``Telescope.trace``
    return rays in the **camera-local** frame so they can be fed straight
    into ``Camera.collect`` / ``Camera.image``.

    **Liveness vs throughput.** IACTrace keeps the two ways a ray can be
    "lost" on separate axes:

    * ``alive`` (bool) answers *"is this a valid, still-propagating
      ray?"*. It is flipped off only by **geometry / occlusion** loss: a
      ray that misses every element in a stage, lands outside an aperture,
      is blocked by an obstruction, or misses the sensor. Once ``False``
      it stays ``False`` (an absorbing state), and the ``origins`` /
      ``directions`` of a dead ray are meaningless — always mask geometry
      with ``alive`` before reading positions.
    * ``values`` (float >= 0) is the **radiometric throughput** of a live
      ray. Every *physical* coefficient multiplies into it — primary
      sampling weight, reflectivity / transmittance, quantum efficiency,
      concentrator throughput. A live ray may legitimately reach ``0``
      (a perfectly absorbing coating, total internal reflection); that is
      distinct from a dead ray and is *not* recorded on the ``alive`` axis.

    As an invariant a dead ray always carries ``values == 0``, so the
    image / response-matrix sums (which add ``values``) need no masking;
    the ``alive`` flag exists so per-ray consumers can tell *why* a ray is
    dark. "Carries light" is simply ``alive & (values > 0)``.

    By the time the bundle reaches ``Camera.collect`` the entries of
    ``values`` are photoelectrons, not raw photons.

    Attributes:
        origins: Ray positions in 3D (n_rays, 3). Meaningful only where
            ``alive`` is ``True``.
        directions: Ray direction vectors (n_rays, 3). Meaningful only
            where ``alive`` is ``True``.
        values: Throughput-weighted ray intensities (n_rays,).
        path_length: Accumulated **optical** path length per ray
            (n_rays,), in metres.
        n: Per-ray refractive index of the medium each ray is
            currently propagating in (n_rays,). Carried so downstream
            consumers (sensor intersection, focal-surface analysis)
            can weight the final geometric leg correctly.
        alive: Per-ray liveness flag (n_rays,), boolean. ``True`` for a
            valid, still-propagating ray. Defaults to all-``True`` at
            construction, i.e. a freshly built bundle is fully alive.
    """

    origins: Array
    directions: Array
    values: Array
    path_length: Array
    n: Array
    alive: Array

    def __init__(
        self,
        origins: Array,
        directions: Array,
        values: Array,
        path_length: Array,
        n: Array,
        alive: Array | None = None,
    ) -> None:
        self.origins = origins
        self.directions = directions
        self.values = values
        self.path_length = path_length
        self.n = n
        self.alive = (
            jnp.ones(values.shape[0], dtype=bool)
            if alive is None
            else jnp.asarray(alive, dtype=bool)
        )

    def replace(self, **changes: Array) -> RayBundle:
        """Copy with the given fields replaced (functional update).

        ``rays.replace(values=v)`` reads better than re-listing all six
        fields; unknown field names raise ``TypeError``.
        """
        fields = {
            "origins": self.origins,
            "directions": self.directions,
            "values": self.values,
            "path_length": self.path_length,
            "n": self.n,
            "alive": self.alive,
        }
        unknown = set(changes) - set(fields)
        if unknown:
            raise TypeError(f"RayBundle has no field(s) {sorted(unknown)}")
        fields.update(changes)
        return RayBundle(**fields)

    def to_frame(self, origin: Array, rotation: Array) -> RayBundle:
        """Express these rays in the local frame given by ``origin`` + Euler ``rotation``.

        ``origin`` is the new frame's position in the current frame;
        ``rotation`` are XYZ Euler angles in degrees.

        This is a **pure coordinate transform**: it moves ``origins`` and
        ``directions`` and leaves ``values`` / ``path_length`` / ``n`` /
        ``alive`` untouched.
        """
        rot = euler_to_matrix(rotation)
        return RayBundle(
            origins=(self.origins - origin) @ rot,
            directions=self.directions @ rot,
            values=self.values,
            path_length=self.path_length,
            n=self.n,
            alive=self.alive,
        )


class LazyRayBundle(eqx.Module):
    """A :class:`RayBundle` described by a render that hasn't run yet.

    Self-contained: holds the optics, obstructions, camera frame, and
    source description needed to evaluate itself. Output
    is delivered in the local frame defined by ``camera_position`` and
    ``camera_rotation``.

    Two ways to consume a :class:`LazyRayBundle`:

    * :meth:`fold`: walk per primary-mirror element with an accumulator,
      so the full ``(n_elements * n_sources * n_samples,)`` ray buffer
      is never materialised. The fused path used by
      :meth:`Camera.image` and :meth:`Camera.response_matrix`.
    * :meth:`materialise`: run the render eagerly and return a flat
      :class:`RayBundle`. Use when the per-ray output itself is the
      result (spot diagrams, :meth:`Camera.collect`).
    """

    optical_groups: list
    obstruction_groups: list
    camera_position: Array
    camera_rotation: Array
    sources: Array
    source_values: Array
    source_type: Literal["point", "parallel"] = eqx.field(static=True)

    def fold(self, accumulator, init):
        """Per-element scan: ``accumulator(carry, rb_local) -> carry``.

        ``rb_local`` is one element's :class:`RayBundle` already
        transformed into the local frame.
        """
        from .render import apply_final_leg_shadow, render_optics_accumulate

        origin, rotation = self.camera_position, self.camera_rotation
        obstructions = self.obstruction_groups

        def in_local_frame(carry, rb_world):
            # Handoff = shadow the final leg (explicit), then a pure reframe.
            rb_world = apply_final_leg_shadow(rb_world, obstructions, origin, rotation)
            return accumulator(carry, rb_world.to_frame(origin, rotation))

        return render_optics_accumulate(
            self.optical_groups,
            self.obstruction_groups,
            self.sources,
            self.source_values,
            self.source_type,
            in_local_frame,
            init,
        )

    def materialise(self) -> RayBundle:
        """Run the render eagerly; return a flat local-frame :class:`RayBundle`."""
        from .render import apply_final_leg_shadow, render_optics

        rb_world = render_optics(
            self.optical_groups,
            self.obstruction_groups,
            self.sources,
            self.source_values,
            self.source_type,
        )
        # Handoff = shadow the final leg (explicit), then a pure reframe.
        rb_world = apply_final_leg_shadow(
            rb_world,
            self.obstruction_groups,
            self.camera_position,
            self.camera_rotation,
        )
        return rb_world.to_frame(self.camera_position, self.camera_rotation)
