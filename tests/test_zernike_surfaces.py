"""Tests for the first-class surface refactor: Zernike and Sum surfaces.

Covers:
- the Noll Zernike basis (normalization, orthogonality, smoothness at origin),
- the standalone ``ZernikeSurfaceGroup`` (sag, decenter, autodiff normal,
  intersection, slicing, validation),
- the composable ``SumSurfaceGroup`` (additivity, equivalence, intersection),
- a regression check that the refactored ``AsphericSurfaceGroup`` still matches
  the module-level reference functions, and
- end-to-end use through ``OpticalElementGroup`` (both render entry points).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import (
    N_ZERNIKE,
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
    compute_sag_and_normal,
    sag,
    zernike_terms,
)


def _fd_slope(sag_fn, x, y, h=1e-5):
    """Central-difference (dz/dx, dz/dy) of a scalar sag function."""
    dzdx = (sag_fn(x + h, y) - sag_fn(x - h, y)) / (2 * h)
    dzdy = (sag_fn(x, y + h) - sag_fn(x, y - h)) / (2 * h)
    return float(dzdx), float(dzdy)


class TestZernikeTerms:
    """The Noll Zernike basis itself."""

    def test_piston_is_unity(self):
        u = jnp.array([0.0, 0.3, -0.7])
        v = jnp.array([0.0, -0.4, 0.1])
        terms = zernike_terms(u, v)
        assert jnp.allclose(terms[..., 0], 1.0)

    def test_named_term_values(self):
        s6 = float(jnp.sqrt(6.0))
        s3 = float(jnp.sqrt(3.0))
        # Z6 = sqrt6 (u^2 - v^2): +s6 at (1,0), -s6 at (0,1)
        assert float(zernike_terms(jnp.array(1.0), jnp.array(0.0))[5]) == pytest.approx(s6)
        assert float(zernike_terms(jnp.array(0.0), jnp.array(1.0))[5]) == pytest.approx(-s6)
        # Z5 = 2 sqrt6 u v
        assert float(zernike_terms(jnp.array(1.0), jnp.array(1.0))[4]) == pytest.approx(2 * s6)
        # Z4 = sqrt3 (2 r^2 - 1): -sqrt3 at origin
        assert float(zernike_terms(jnp.array(0.0), jnp.array(0.0))[3]) == pytest.approx(-s3)
        # Z2 = 2u, Z3 = 2v
        assert float(zernike_terms(jnp.array(0.5), jnp.array(0.0))[1]) == pytest.approx(1.0)
        assert float(zernike_terms(jnp.array(0.0), jnp.array(0.5))[2]) == pytest.approx(1.0)

    def test_rms_normalized(self):
        """Each term has unit RMS over the unit disk (Noll normalization)."""
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        # Rejection-sample points uniformly in the unit disk.
        u = jax.random.uniform(k1, (200000,), minval=-1.0, maxval=1.0)
        v = jax.random.uniform(k2, (200000,), minval=-1.0, maxval=1.0)
        inside = (u * u + v * v) <= 1.0
        terms = zernike_terms(u, v)  # (M, 11)
        ms = jnp.sum(jnp.where(inside[:, None], terms**2, 0.0), axis=0) / jnp.sum(inside)
        # Unit RMS -> mean square ~ 1 for every term.
        assert jnp.allclose(ms, 1.0, atol=0.03), ms

    def test_orthogonality(self):
        """Distinct Noll terms are orthogonal over the unit disk."""
        key = jax.random.key(1)
        k1, k2 = jax.random.split(key)
        u = jax.random.uniform(k1, (200000,), minval=-1.0, maxval=1.0)
        v = jax.random.uniform(k2, (200000,), minval=-1.0, maxval=1.0)
        inside = (u * u + v * v) <= 1.0
        terms = jnp.where(inside[:, None], zernike_terms(u, v), 0.0)
        gram = (terms.T @ terms) / jnp.sum(inside)
        off = gram - jnp.diag(jnp.diag(gram))
        assert jnp.max(jnp.abs(off)) < 0.03, jnp.max(jnp.abs(off))

    def test_smooth_at_origin(self):
        """Gradient is finite at the origin (Cartesian forms, no atan2)."""
        def f(x, y):
            return jnp.sum(zernike_terms(x, y))
        gx = jax.grad(f, 0)(0.0, 0.0)
        gy = jax.grad(f, 1)(0.0, 0.0)
        assert jnp.isfinite(gx) and jnp.isfinite(gy)


def _zernike(coeffs_row, r_norm, offset=(0.0, 0.0)):
    """Build a single-element ZernikeSurfaceGroup."""
    return ZernikeSurfaceGroup(
        coeffs=jnp.asarray([coeffs_row]),
        r_norm=jnp.asarray([r_norm]),
        offsets=jnp.asarray([offset]),
    )


class TestZernikeSurfaceGroup:
    def test_sag_zero_at_origin(self):
        """Re-zero: sag at the decenter point is zero for any coefficients."""
        zg = _zernike([0.0, 0.1, -0.2, 0.3, 0.05, -0.05], r_norm=2.0)
        assert float(zg.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-12)

    def test_defocus_is_paraboloid(self):
        """Pure defocus term gives z = 2*sqrt3*c4*(x^2+y^2)/r_norm^2."""
        c4, R = 1e-3, 2.0
        zg = _zernike([0.0, 0.0, 0.0, c4], r_norm=R)
        for x, y in [(0.5, 0.0), (0.3, -0.7), (1.0, 1.0)]:
            expected = 2.0 * np.sqrt(3.0) * c4 * (x * x + y * y) / (R * R)
            assert float(zg.sag_at(0, x, y)) == pytest.approx(expected, rel=1e-6)

    def test_normal_matches_finite_difference(self):
        coeffs = [0.0, 1e-3, -2e-3, 1.5e-3, 8e-4, -6e-4, 3e-4, -2e-4]
        zg = _zernike(coeffs, r_norm=1.5)
        elem = zg._index(0)
        for x, y in [(0.4, 0.2), (-0.6, 0.3), (0.9, -0.5)]:
            _, normal = elem.compute_sag_and_normal_at(x, y)
            dzdx, dzdy = _fd_slope(lambda xx, yy: elem._sag_local(xx, yy), x, y)
            expected = np.array([-dzdx, -dzdy, 1.0])
            expected = expected / np.linalg.norm(expected)
            assert np.allclose(np.asarray(normal), expected, atol=1e-4)
            assert float(jnp.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-10)

    def test_standalone_intersection_on_surface(self):
        """A standalone Zernike surface (no asphere) intersects correctly."""
        zg = _zernike([0.0, 0.0, 0.0, 2e-3, 1e-3, -1e-3], r_norm=1.0)
        origin = jnp.array([0.3, -0.2, 5.0])
        direction = jnp.array([0.02, -0.01, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, normal = zg.intersect_at(0, origin, direction)
        assert jnp.isfinite(t)
        hit = origin + t * direction
        # Hit lies on the ray and on the surface.
        assert np.allclose(np.asarray(point), np.asarray(hit), atol=1e-6)
        z_surf = zg.sag_at(0, float(hit[0]), float(hit[1]))
        assert float(hit[2]) == pytest.approx(float(z_surf), abs=1e-6)

    def test_decenter_shifts_sampling(self):
        """A decentered Zernike samples the map off-centre but stays re-zeroed."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 1e-3, 2e-3]  # astigmatism only
        offset = (0.4, -0.3)
        zg = _zernike(coeffs, r_norm=1.0, offset=offset)
        centered = _zernike(coeffs, r_norm=1.0, offset=(0.0, 0.0))
        # sag at decenter point is still zero
        assert float(zg.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
        # decentered sag(x,y) == centered intrinsic shifted and re-zeroed
        x, y = 0.2, 0.1
        ox, oy = offset
        expected = (
            float(centered._index(0)._sag_intrinsic(x + ox, y + oy))
            - float(centered._index(0)._sag_intrinsic(ox, oy))
        )
        assert float(zg.sag_at(0, x, y)) == pytest.approx(expected, rel=1e-6, abs=1e-12)

    def test_index_slicing(self):
        zg = ZernikeSurfaceGroup(
            coeffs=jnp.array([[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0]]),
            r_norm=jnp.array([1.0, 2.0]),
        )
        elem = zg._index(1)
        assert np.allclose(np.asarray(elem.coeffs), [0.0, 4.0, 5.0, 6.0])
        assert float(elem.r_norm) == pytest.approx(2.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            ZernikeSurfaceGroup(coeffs=jnp.zeros((4,)), r_norm=jnp.ones(1))  # not 2D
        with pytest.raises(ValueError):
            ZernikeSurfaceGroup(coeffs=jnp.zeros((1, N_ZERNIKE + 1)), r_norm=jnp.ones(1))


def _asphere(curv, conic=0.0, aspheric=None, offset=(0.0, 0.0)):
    asph = jnp.zeros((1, 0)) if aspheric is None else jnp.asarray([aspheric])
    return AsphericSurfaceGroup(
        curvatures=jnp.asarray([curv]),
        conics=jnp.asarray([conic]),
        aspherics=asph,
        offsets=jnp.asarray([offset]),
    )


class TestAsphericRegression:
    """The refactored AsphericSurfaceGroup matches the reference functions."""

    def test_sag_and_normal_match_module_functions(self):
        c, k = 0.08, -1.0
        a = jnp.array([1e-4, -2e-6])
        offset = jnp.array([0.3, -0.2])
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([c]),
            conics=jnp.array([k]),
            aspherics=jnp.asarray([a]),
            offsets=jnp.asarray([offset]),
        )
        elem = asph._index(0)
        for x, y in [(0.0, 0.0), (0.5, 0.3), (-0.7, 0.2)]:
            z_ref = float(sag(x, y, offset, c, k, a))
            assert float(asph.sag_at(0, x, y)) == pytest.approx(z_ref, rel=1e-7, abs=1e-12)
            pt, nrm = elem.compute_sag_and_normal_at(x, y)
            pt_ref, nrm_ref = compute_sag_and_normal(x, y, offset, c, k, a)
            assert np.allclose(np.asarray(pt), np.asarray(pt_ref), atol=1e-9)
            assert np.allclose(np.asarray(nrm), np.asarray(nrm_ref), atol=1e-9)

    def test_intersection_lands_on_surface(self):
        asph = _asphere(0.1, conic=-1.0)
        origin = jnp.array([2.0, 1.0, 12.0])
        direction = jnp.array([-0.05, 0.02, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, _ = asph.intersect_at(0, origin, direction)
        z_surf = asph.sag_at(0, float(point[0]), float(point[1]))
        assert float(point[2]) == pytest.approx(float(z_surf), abs=1e-6)


class TestSumSurfaceGroup:
    def test_single_component_equals_component(self):
        asph = _asphere(0.1, conic=-1.0, aspheric=[1e-4])
        s = SumSurfaceGroup([asph])
        for x, y in [(0.0, 0.0), (0.4, 0.3), (-0.6, 0.1)]:
            assert float(s.sag_at(0, x, y)) == pytest.approx(
                float(asph.sag_at(0, x, y)), rel=1e-7, abs=1e-12
            )

    def test_additive(self):
        """Sum sag = component sags added (components with zero offset)."""
        asph = _asphere(0.05, conic=0.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4], r_norm=1.0)
        s = SumSurfaceGroup([asph, zg])
        for x, y in [(0.3, 0.2), (-0.5, 0.4), (0.8, -0.1)]:
            expected = float(asph.sag_at(0, x, y)) + float(zg.sag_at(0, x, y))
            assert float(s.sag_at(0, x, y)) == pytest.approx(expected, rel=1e-6, abs=1e-12)

    def test_zero_zernike_equals_asphere(self):
        asph = _asphere(0.07, conic=-1.0)
        zg = _zernike([0.0] * 6, r_norm=1.0)
        s = SumSurfaceGroup([asph, zg])
        origin = jnp.array([1.5, -0.8, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t_s, p_s, n_s = s.intersect_at(0, origin, direction)
        t_a, p_a, n_a = asph.intersect_at(0, origin, direction)
        assert float(t_s) == pytest.approx(float(t_a), rel=1e-6)
        assert np.allclose(np.asarray(p_s), np.asarray(p_a), atol=1e-6)
        assert np.allclose(np.asarray(n_s), np.asarray(n_a), atol=1e-6)

    def test_normal_matches_finite_difference(self):
        asph = _asphere(0.05, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4, 3e-4], r_norm=1.2)
        s = SumSurfaceGroup([asph, zg])
        elem = s._index(0)
        for x, y in [(0.4, 0.2), (-0.6, 0.35)]:
            _, normal = elem.compute_sag_and_normal_at(x, y)
            dzdx, dzdy = _fd_slope(lambda xx, yy: elem._sag_local(xx, yy), x, y)
            expected = np.array([-dzdx, -dzdy, 1.0])
            expected = expected / np.linalg.norm(expected)
            assert np.allclose(np.asarray(normal), expected, atol=1e-4)
            assert float(jnp.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-10)

    def test_intersection_on_surface(self):
        asph = _asphere(0.06, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 0.0, 2e-3, -1e-3, 5e-4], r_norm=1.0)
        s = SumSurfaceGroup([asph, zg])
        origin = jnp.array([0.5, -0.3, 8.0])
        direction = jnp.array([0.03, 0.01, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, _ = s.intersect_at(0, origin, direction)
        hit = origin + t * direction
        assert np.allclose(np.asarray(point), np.asarray(hit), atol=1e-6)
        z_surf = s.sag_at(0, float(hit[0]), float(hit[1]))
        assert float(hit[2]) == pytest.approx(float(z_surf), abs=1e-6)

    def test_index_slicing_recurses(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.1, 0.2]),
            conics=jnp.zeros(2),
            aspherics=jnp.zeros((2, 0)),
            offsets=jnp.zeros((2, 2)),
        )
        zg = ZernikeSurfaceGroup(
            coeffs=jnp.array([[0.0, 1.0, 2.0, 3.0], [0.0, 4.0, 5.0, 6.0]]),
            r_norm=jnp.array([1.0, 2.0]),
        )
        s = SumSurfaceGroup([asph, zg])
        elem = s._index(1)
        assert float(elem.components[0].curvatures) == pytest.approx(0.2)
        assert np.allclose(np.asarray(elem.components[1].coeffs), [0.0, 4.0, 5.0, 6.0])

    def test_validation(self):
        with pytest.raises(ValueError):
            SumSurfaceGroup([])
        # Mismatched element counts (N=1 asphere vs N=2 Zernike).
        zg_n2 = ZernikeSurfaceGroup(coeffs=jnp.zeros((2, 4)), r_norm=jnp.ones(2))
        with pytest.raises(ValueError):
            SumSurfaceGroup([_asphere(0.1), zg_n2])


def _mirror_group_with_surface(surface, radius=0.5):
    """Wrap a surface in a single-element disk mirror OpticalElementGroup."""
    n = surface.offsets.shape[0]
    aperture = DiskAperture(radii=jnp.full(n, radius), inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=0,
        n_samples=64,
    )


class TestRenderPipeline:
    """Both render entry points accept the new surfaces (first-class)."""

    def test_transform_to_world_zernike(self):
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4], r_norm=0.5)
        group = _mirror_group_with_surface(zg)
        points, normals, weights = group.transform_to_world()
        assert points.shape == (1, 64, 3)
        assert jnp.all(jnp.isfinite(points))
        assert jnp.all(jnp.isfinite(normals))
        assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)

    def test_transform_to_world_sum(self):
        asph = _asphere(0.08, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4], r_norm=0.5)
        group = _mirror_group_with_surface(SumSurfaceGroup([asph, zg]))
        points, normals, _ = group.transform_to_world()
        assert jnp.all(jnp.isfinite(points))
        assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)

    def test_intersect_at_vmapped_sum(self):
        asph = _asphere(0.08, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4], r_norm=0.5)
        group = _mirror_group_with_surface(SumSurfaceGroup([asph, zg]))
        n_rays = 32
        key = jax.random.key(7)
        xy = jax.random.uniform(key, (n_rays, 2), minval=-0.3, maxval=0.3)
        origins = jnp.concatenate([xy, jnp.full((n_rays, 1), 6.0)], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        ts, pts, norms = jax.vmap(
            lambda o, d: group.surface.intersect_at(0, o, d)
        )(origins, directions)
        assert jnp.all(jnp.isfinite(ts))
        # Each hit lies on the surface.
        z_surf = jax.vmap(lambda p: group.surface.sag_at(0, p[0], p[1]))(pts)
        assert jnp.allclose(pts[:, 2], z_surf, atol=1e-6)
