from __future__ import annotations

from typing import Literal

import equinox as eqx
from jax import Array

from .transforms import euler_to_matrix


class RayBundle(eqx.Module):
    """Bundle of rays through the optical system.

    Carries ray positions, directions, weights, and path lengths.

    The frame of ``origins`` / ``directions`` is implicit and depends on
    where the bundle came from: ``Telescope.render`` and ``Telescope.trace``
    return rays in the **camera-local** frame so they can be fed straight
    into ``Camera.collect`` / ``Camera.image``.

    ``values`` is a dimensionless throughput-weighted photon count.
    Every interaction along the optical path multiplies into it:

        primary sampling weight  ->  reflectivity / refractivity ->
        aperture mask  ->  obstruction shadow  ->  concentrator throughput
        ->  quantum efficiency

    so by the time the bundle reaches ``Camera.collect`` the entries of
    ``values`` are photoelectrons, not raw photons.

    Attributes:
        origins: Ray positions in 3D (n_rays, 3)
        directions: Ray direction vectors (n_rays, 3)
        values: Throughput-weighted ray intensities (n_rays,)
        path_length: Accumulated **optical** path length per ray
            (n_rays,), in metres.
        n: Per-ray refractive index of the medium each ray is
            currently propagating in (n_rays,). Carried so downstream
            consumers (sensor intersection, focal-surface analysis)
            can weight the final geometric leg correctly. Slab
            interactions additionally contribute their internal
            ``n_in * L_internal`` term to ``path_length``.
    """

    origins: Array
    directions: Array
    values: Array
    path_length: Array
    n: Array

    def to_frame(self, origin: Array, rotation: Array) -> RayBundle:
        """Express these rays in the local frame given by ``origin`` + Euler ``rotation``.

        ``origin`` is the new frame's position in the current frame;
        ``rotation`` are XYZ Euler angles in degrees.
        """
        rot = euler_to_matrix(rotation)
        return RayBundle(
            origins=(self.origins - origin) @ rot,
            directions=self.directions @ rot,
            values=self.values,
            path_length=self.path_length,
            n=self.n,
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
        from .render import render_optics_accumulate
        origin, rotation = self.camera_position, self.camera_rotation

        def in_local_frame(carry, rb_world):
            return accumulator(carry, rb_world.to_frame(origin, rotation))

        return render_optics_accumulate(
            self.optical_groups, self.obstruction_groups,
            self.sources, self.source_values, self.source_type,
            in_local_frame, init,
        )

    def materialise(self) -> RayBundle:
        """Run the render eagerly; return a flat local-frame :class:`RayBundle`."""
        from .render import render_optics
        rb_world = render_optics(
            self.optical_groups, self.obstruction_groups,
            self.sources, self.source_values, self.source_type,
        )
        return rb_world.to_frame(self.camera_position, self.camera_rotation)
