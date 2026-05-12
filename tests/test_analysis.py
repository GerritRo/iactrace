import jax.numpy as jnp
import pytest

from iactrace import RayBundle
from iactrace.analysis import (
    AsphericFocalSurface,
    FlatFocalPlane,
    FocalSurfaceHits,
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
    )


class TestFlatFocalPlane:
    """Basic intersection behaviour of FlatFocalPlane."""

    def test_returns_focal_surface_hits(self):
        plane = FlatFocalPlane()
        bundle = _downward_bundle([[0.0, 0.0, 1.0]])
        hits = plane.intersect(bundle)
        assert isinstance(hits, FocalSurfaceHits)

    def test_downward_ray_hits_at_xy_origin(self):
        """A ray on the optical axis hits at (0, 0) at unit distance."""
        plane = FlatFocalPlane()
        bundle = _downward_bundle([[0.0, 0.0, 1.0]])
        hits = plane.intersect(bundle)

        assert bool(hits.hit_mask[0])
        assert jnp.allclose(hits.xy_local[0], jnp.zeros(2), atol=1e-6)
        assert jnp.isclose(hits.z_local[0], 0.0, atol=1e-6)
        assert jnp.isclose(hits.t[0], 1.0, atol=1e-6)

    def test_xy_offset_preserved(self):
        """Off-axis ray preserves its (x, y) at the hit."""
        plane = FlatFocalPlane()
        bundle = _downward_bundle([[0.3, -0.2, 2.0]])
        hits = plane.intersect(bundle)

        assert bool(hits.hit_mask[0])
        assert jnp.allclose(hits.xy_local[0], jnp.array([0.3, -0.2]), atol=1e-6)
        assert jnp.isclose(hits.t[0], 2.0, atol=1e-6)

    def test_translated_plane(self):
        """Plane at z=0.5 is hit at t=0.5 by a ray starting at z=1."""
        plane = FlatFocalPlane(position=jnp.array([0.0, 0.0, 0.5]))
        bundle = _downward_bundle([[0.0, 0.0, 1.0]])
        hits = plane.intersect(bundle)

        assert bool(hits.hit_mask[0])
        assert jnp.isclose(hits.t[0], 0.5, atol=1e-6)


class TestAsphericFocalSurface:
    """Basic intersection behaviour of AsphericFocalSurface."""

    def test_zero_curvature_matches_flat_plane(self):
        """With c=0 and no aspherics, the surface degenerates to a plane."""
        flat = FlatFocalPlane()
        aspheric = AsphericFocalSurface(curvature=0.0)
        bundle = _downward_bundle([
            [0.0, 0.0, 1.0],
            [0.1, -0.05, 1.0],
        ])

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
        # For a sphere with R=2, sag at r=0.4 is 2 - sqrt(2^2 - 0.4^2) ≈ 0.0404
        expected_sag = 2.0 - jnp.sqrt(4.0 - 0.16)
        assert jnp.isclose(hits.z_local[0], expected_sag, atol=1e-4)


class TestVmapping:
    """The intersect path is vmapped — ensure multi-ray bundles work."""

    def test_multiple_rays_independent(self):
        plane = FlatFocalPlane()
        bundle = _downward_bundle([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0],
        ])
        hits = plane.intersect(bundle)

        assert hits.hit_mask.shape == (3,)
        assert hits.xy_local.shape == (3, 2)
        assert jnp.all(hits.hit_mask)
        assert jnp.allclose(
            hits.xy_local,
            jnp.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]),
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
