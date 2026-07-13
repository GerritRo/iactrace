import jax.numpy as jnp

from iactrace import RayBundle
from iactrace.analysis import (
    AsphericFocalSurface,
    FlatFocalPlane,
)


def _downward_bundle(origins):
    """Rays pointing along -z with unit intensity and zero path-length."""
    origins = jnp.asarray(origins, dtype=float)
    n = origins.shape[0]
    directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n, 3))
    return RayBundle(
        origins=origins,
        directions=directions,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


class TestFlatFocalPlane:
    """Basic intersection behaviour of FlatFocalPlane."""

    def test_intersection_axis_offset_and_translation(self):
        """On-axis and off-axis rays hit at their (x, y) at the right distance,
        and translating the plane moves the hit distance accordingly."""
        plane = FlatFocalPlane()
        # on-axis ray from z=1 hits (0, 0) at t=1
        on_axis = plane.intersect(_downward_bundle([[0.0, 0.0, 1.0]]))
        assert bool(on_axis.hit_mask[0])
        assert jnp.allclose(on_axis.xy_local[0], jnp.zeros(2), atol=1e-6)
        assert jnp.isclose(on_axis.z_local[0], 0.0, atol=1e-6)
        assert jnp.isclose(on_axis.t[0], 1.0, atol=1e-6)
        # off-axis ray preserves its (x, y)
        off_axis = plane.intersect(_downward_bundle([[0.3, -0.2, 2.0]]))
        assert jnp.allclose(off_axis.xy_local[0], jnp.array([0.3, -0.2]), atol=1e-6)
        assert jnp.isclose(off_axis.t[0], 2.0, atol=1e-6)
        # a plane translated to z=0.5 is hit at t=0.5 by a ray from z=1
        translated = FlatFocalPlane(position=jnp.array([0.0, 0.0, 0.5])).intersect(
            _downward_bundle([[0.0, 0.0, 1.0]])
        )
        assert jnp.isclose(translated.t[0], 0.5, atol=1e-6)


class TestAsphericFocalSurface:
    """Basic intersection behaviour of AsphericFocalSurface."""

    def test_zero_curvature_matches_flat_plane(self):
        """With c=0 and no aspherics, the surface degenerates to a plane."""
        flat = FlatFocalPlane()
        aspheric = AsphericFocalSurface(curvature=0.0)
        bundle = _downward_bundle(
            [
                [0.0, 0.0, 1.0],
                [0.1, -0.05, 1.0],
            ]
        )

        flat_hits = flat.intersect(bundle)
        asph_hits = aspheric.intersect(bundle)

        assert jnp.all(asph_hits.hit_mask == flat_hits.hit_mask)
        assert jnp.allclose(asph_hits.xy_local, flat_hits.xy_local, atol=1e-5)
        assert jnp.allclose(asph_hits.t, flat_hits.t, atol=1e-5)

    def test_curved_surface_sag_nonzero(self):
        """A non-zero curvature produces a non-zero sag for off-axis rays."""
        # 1/R = 0.5 -> R = 2 m
        aspheric = AsphericFocalSurface(curvature=0.5, conic=0.0)
        bundle = _downward_bundle([[0.4, 0.0, 5.0]])
        hits = aspheric.intersect(bundle)

        assert bool(hits.hit_mask[0])
        # For a sphere with R=2, sag at r=0.4 is 2 - sqrt(2^2 - 0.4^2) ~ 0.0404
        expected_sag = 2.0 - jnp.sqrt(4.0 - 0.16)
        assert jnp.isclose(hits.z_local[0], expected_sag, atol=1e-4)
