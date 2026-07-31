import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace.core.bsdf import GaussianBSDF, _apply_perturbation
from iactrace.core.intersections import intersect_conic, newton_raphson_intersect
from iactrace.core.surfaces import (
    N_ZERNIKE,
    AsphericSurfaceGroup,
    FreeformSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
    bicubic_interp,
    compute_sag_and_normal,
    sag,
    sag_raw,
    zernike_terms,
)

from ._helpers import fd_slope, mirror_group_with_surface

# =============================================================================
# Aspheric / conic surfaces + intersectors
# =============================================================================


class TestSurfaceSag:
    """Test surface sag calculations for known cases."""

    def test_flat_and_offset_rezero(self):
        """Flat surface has zero sag everywhere; sag() with an offset returns 0
        at the offset point itself."""
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)]:
            z = sag_raw(x, y, curvature=0.0, conic=0.0, aspheric=jnp.array([]))
            assert jnp.allclose(z, 0.0, atol=1e-12)
        offset = jnp.array([2.0, 3.0])
        assert jnp.allclose(sag(0.0, 0.0, offset, 0.05, 0.0, jnp.array([])), 0.0, atol=1e-12)

    def test_spherical_sag_formula(self):
        """Spherical sag matches z = c*r^2 / (1 + sqrt(1 - c^2*r^2)) and depends
        only on r^2 (rotational symmetry)."""
        c, r = 0.02, 3.0
        z = sag_raw(r, 0.0, c, 0.0, jnp.array([]))
        z_expected = c * r**2 / (1 + jnp.sqrt(1 - c * c * r**2))
        assert jnp.allclose(z, z_expected, atol=1e-12)
        # symmetric in r: same value along x, along y, and at 45 degrees
        z_y = sag_raw(0.0, r, c, 0.0, jnp.array([]))
        z_diag = sag_raw(r / jnp.sqrt(2), r / jnp.sqrt(2), c, 0.0, jnp.array([]))
        assert jnp.allclose(z, z_y, atol=1e-12)
        assert jnp.allclose(z, z_diag, atol=1e-12)

    def test_paraboloid_sag_formula(self):
        """Parabolic sag: z = c*r^2 / 2 for paraboloid (k=-1)."""
        c, r = 0.02, 3.0
        z_computed = sag_raw(r, 0.0, c, conic=-1.0, aspheric=jnp.array([]))
        assert jnp.allclose(z_computed, c * r * r / 2.0, atol=1e-12)

    def test_aspheric_powers_are_consecutive_even(self):
        """aspheric[i] must multiply r^(2*i+4): r^4, r^6, r^8, r^10, ..."""
        r = 0.7
        coeffs = jnp.array([1.1, -2.3, 3.7, -4.2])  # r^4, r^6, r^8, r^10
        z_computed = sag_raw(r, 0.0, curvature=0.0, conic=0.0, aspheric=coeffs)
        z_expected = coeffs[0] * r**4 + coeffs[1] * r**6 + coeffs[2] * r**8 + coeffs[3] * r**10
        assert jnp.allclose(z_computed, z_expected, atol=1e-12)


class TestSurfaceNormals:
    """Test that computed surface normals are correct."""

    def test_normal_is_unit_length_and_flat_is_z(self):
        """Surface normal is always unit length; a flat surface (c=0) points +z."""
        offset = jnp.array([0.0, 0.0])
        for c in [0.0, 0.01, 0.1]:
            for x, y in [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)]:
                _, normal = compute_sag_and_normal(x, y, offset, c, 0.0, jnp.array([]))
                assert jnp.allclose(jnp.linalg.norm(normal), 1.0, atol=1e-10)
                if c == 0.0:
                    assert jnp.allclose(normal, jnp.array([0.0, 0.0, 1.0]), atol=1e-8)

    def test_spherical_normal_points_to_center(self):
        """For a spherical surface, normal at (x,y,z) points toward center of curvature."""
        c = 0.05  # curvature = 1/R, so R = 20
        R = 1.0 / c
        offset = jnp.array([0.0, 0.0])
        x, y = 2.0, 0.0
        point, normal = compute_sag_and_normal(x, y, offset, c, 0.0, jnp.array([]))
        center = jnp.array([0.0, 0.0, R])
        to_center = center - point
        to_center = to_center / jnp.linalg.norm(to_center)
        assert jnp.allclose(jnp.abs(jnp.dot(normal, to_center)), 1.0, atol=1e-6)


class TestPerturbation:
    """Test surface roughness perturbation via BSDF modules."""

    def test_zero_scale_no_perturbation(self):
        """With zero perturbation scale, normals are unchanged."""
        normals = jnp.array([[0.0, 0.0, 1.0], [0.1, 0.2, 0.97]])
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
        angles = jnp.array([[0.5, -0.3], [1.0, 0.2]])
        perturbed = _apply_perturbation(normals, angles, scale=0.0)
        assert jnp.allclose(perturbed, normals, atol=1e-10)

    def test_perturbed_normals_unit_and_small(self):
        """Perturbed normals stay unit vectors, and a small scale gives a small
        angular deviation."""
        key = jax.random.key(42)
        normals = jnp.array([[0.0, 0.0, 1.0], [0.1, 0.2, 0.97], [-0.3, 0.1, 0.95]])
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
        bsdf = GaussianBSDF(scale=jnp.array([0.01, 0.01, 0.01]))
        perturbed = bsdf.perturb_normals(normals, key, jnp.array([0, 1, 2]))
        assert jnp.allclose(jnp.linalg.norm(perturbed, axis=-1), 1.0, atol=1e-10)

        tiny = GaussianBSDF(scale=jnp.array([1e-4]))
        normal = jnp.array([[0.0, 0.0, 1.0]])
        p = tiny.perturb_normals(normal, jax.random.key(123), jnp.array([0]))
        assert jnp.sum(normal * p) > 0.9999


class TestConicIntersection:
    """Ray-conic surface intersection for flat, parabolic, and hyperbolic conics."""

    def _residual(self, point, c, k):
        """c*(x^2+y^2) + (1+k)*c*z^2 - 2z = 0 on the conic surface."""
        x, y, z = point[0], point[1], point[2]
        return c * (x**2 + y**2) + (1 + k) * c * z**2 - 2 * z

    def test_flat_paraboloid_and_hyperboloid(self):
        """Flat (c=0) is the plane z=0; parabolic and hyperbolic conic hits land
        on the surface."""
        t = intersect_conic(jnp.array([0.0, 0.0, 5.0]), jnp.array([0.0, 0.0, -1.0]), 0.0, 0.0)
        assert jnp.allclose(t, 5.0)
        for c, k, origin, direction in [
            (0.05, -1.0, jnp.array([3.0, 2.0, 15.0]), jnp.array([-0.1, -0.05, -1.0])),
            (0.1, -2.0, jnp.array([0.5, 0.5, 10.0]), jnp.array([0.0, 0.0, -1.0])),
        ]:
            direction = direction / jnp.linalg.norm(direction)
            t = intersect_conic(origin, direction, c, k)
            hit = self._hit(origin, direction, t)
            assert jnp.allclose(self._residual(hit, c, k), 0.0, atol=self._tol(origin))

    def _sag(self, x, y, c, k):
        """The sag branch: z = c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))."""
        r2 = x**2 + y**2
        return r2 * c / (1 + jnp.sqrt(1 - (1 + k) * c * c * r2))

    @staticmethod
    def _hit(origin, direction, t):
        """Step to the hit the well-conditioned way, from the closest approach to
        the vertex. A plain ``origin + t * direction`` cancels two vectors of
        order the source distance, so in float32 the *test's* own reconstruction
        would be noisier than what it is trying to measure."""
        t0 = -jnp.dot(origin, direction)
        return origin + t0 * direction + (t - t0) * direction

    @staticmethod
    def _tol(origin):
        """A few float32 ulps at this ray's magnitude.

        float32 carries ~1.2e-7 relative precision, so the hit is only known to
        ``eps * |origin|``; the conic residual differentiates that with a factor
        of about two (the ``-2z`` term). This stays far tighter than the thing
        under test -- telling the sag branch (z ~ 0.004) from the far sheet
        (z ~ 60) -- at every distance exercised here.
        """
        return 1e-6 * max(1.0, float(jnp.linalg.norm(origin)))

    @pytest.mark.parametrize("z0", [5.0, 30.0, 59.0, 61.0, 100.0, 1000.0])
    def test_hit_is_on_the_sag_branch_at_any_source_distance(self, z0):
        """A sphere's implicit conic is closed -- the ball of radius R centred at
        (0, 0, R), spanning z in [0, 2R]. A ray from beyond 2R crosses the far
        sheet first, so the *nearest* forward root is the wrong one: the hit must
        be the vertex sheet, i.e. z = sag(x, y), at every distance.
        """
        c, k = 1.0 / 30.0, 0.0  # R = 30 -> the far sheet sits at z ~ 60
        origin = jnp.array([0.4, -0.3, z0])
        direction = jnp.array([0.0, 0.0, -1.0])

        t = intersect_conic(origin, direction, c, k)
        hit = self._hit(origin, direction, t)
        atol = self._tol(origin)

        assert jnp.isfinite(t)
        # On the conic at all (the far sheet satisfies this too) ...
        assert jnp.allclose(self._residual(hit, c, k), 0.0, atol=atol)
        # ... and specifically on the sag branch near the vertex.
        assert jnp.allclose(hit[2], self._sag(hit[0], hit[1], c, k), atol=atol)
        assert (1 + k) * c * hit[2] <= 1.0 + 1e-6
        # A vertical ray keeps its transverse position.
        assert jnp.allclose(hit[:2], origin[:2], atol=atol)

    def test_far_sheet_is_never_returned_for_a_tilted_ray(self):
        """Same trap off-axis and tilted, where both roots are forward."""
        c, k = 1.0 / 30.0, 0.0
        origin = jnp.array([2.0, -1.0, 200.0])
        direction = jnp.array([-0.02, 0.01, -1.0])
        direction = direction / jnp.linalg.norm(direction)

        t = intersect_conic(origin, direction, c, k)
        hit = self._hit(origin, direction, t)

        assert jnp.allclose(hit[2], self._sag(hit[0], hit[1], c, k), atol=self._tol(origin))
        assert hit[2] < 30.0, "returned the far sheet of the sphere"


class TestNewtonRaphsonIntersect:
    """Newton-Raphson ray-surface intersection for z = sag(x, y)."""

    def _verify_hit(self, sag_fn, ray_origin, ray_direction, t, hit_xy, tol=1e-6):
        hit_point = ray_origin + t * ray_direction
        assert jnp.allclose(hit_xy[0], hit_point[0], atol=tol)
        assert jnp.allclose(hit_xy[1], hit_point[1], atol=tol)
        assert jnp.allclose(hit_point[2], sag_fn(hit_xy[0], hit_xy[1]), atol=tol)

    def test_paraboloid_hit(self):
        """Paraboloid z = c*(x^2 + y^2) intersection lands on the surface."""
        c = 0.1

        def sag_fn(x, y):
            return c * (x**2 + y**2)

        origin = jnp.array([2.0, 0.0, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        self._verify_hit(sag_fn, origin, direction, t, hit_xy)
        assert jnp.allclose(t, 9.6, atol=1e-6)

    def test_aspheric_hit(self):
        """Aspheric surface with higher-order terms lands on the surface."""
        c, k = 0.1, -1.5
        a4, a6 = 1e-4, 1e-6

        def sag_fn(x, y):
            r2 = x**2 + y**2
            denom = 1 + jnp.sqrt(1 - (1 + k) * c**2 * r2)
            return c * r2 / denom + a4 * r2**2 + a6 * r2**3

        origin = jnp.array([1.0, 1.0, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        self._verify_hit(sag_fn, origin, direction, t, hit_xy)

    def test_missing_rays_return_no_hit(self):
        """Parallel, pointing-away, and behind-the-surface rays all miss."""

        def sag_fn(x, y):
            return 0.0

        # parallel to the plane
        t, _, valid = newton_raphson_intersect(
            sag_fn, jnp.array([0.0, 0.0, 5.0]), jnp.array([1.0, 0.0, 0.0])
        )
        assert not valid or jnp.isinf(t)
        # pointing away from the surface
        t, _, valid = newton_raphson_intersect(
            sag_fn, jnp.array([0.0, 0.0, 5.0]), jnp.array([0.0, 0.0, 1.0])
        )
        assert not valid or jnp.isinf(t)
        # starting behind, travelling further away
        t, _, valid = newton_raphson_intersect(
            sag_fn, jnp.array([0.0, 0.0, -5.0]), jnp.array([0.0, 0.0, -1.0])
        )
        assert not valid
        assert jnp.isinf(t)

    def test_robust_at_large_distance_and_high_curvature(self):
        """Converges at very large t and with high curvature."""

        def flat(x, y):
            return 0.0

        t, _, valid = newton_raphson_intersect(
            flat, jnp.array([0.0, 0.0, 1e6]), jnp.array([0.0, 0.0, -1.0])
        )
        assert valid
        assert jnp.allclose(t, 1e6, rtol=1e-6)

        def steep(x, y):
            return 1.0 * (x**2 + y**2)

        origin = jnp.array([0.5, 0.5, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(steep, origin, direction)
        assert valid
        self._verify_hit(steep, origin, direction, t, hit_xy)


class TestPureConicBypass:
    """AsphericSurfaceGroup with no aspheric terms intersects in closed form.

    ``_t_guess_is_exact`` bypasses the Newton iteration; the result must be
    indistinguishable from the Newton-refined path on the same geometry."""

    @staticmethod
    def _group(aspherics):
        from iactrace.core.surfaces import AsphericSurfaceGroup

        return AsphericSurfaceGroup(
            offsets=jnp.asarray([[0.05, -0.02]]),
            curvatures=jnp.asarray([-1.0 / 12.0]),
            conics=jnp.asarray([-0.4]),
            aspherics=aspherics,
        )

    def test_bypass_matches_newton(self):
        # Same conic, once as a pure conic (closed form) and once with a zero
        # aspheric term (forcing Newton): t, point and normal must agree.
        pure = self._group(jnp.zeros((1, 0)))
        poly = self._group(jnp.zeros((1, 1)))
        key = jax.random.PRNGKey(3)
        n = 500
        o = jnp.concatenate(
            [jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0), jnp.full((n, 1), 10.0)],
            axis=1,
        )
        d = jnp.tile(jnp.array([0.03, -0.01, -1.0]), (n, 1))
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True)

        t1, p1, n1 = jax.vmap(lambda oo, dd: pure.intersect_at(0, oo, dd))(o, d)
        t2, p2, n2 = jax.vmap(lambda oo, dd: poly.intersect_at(0, oo, dd))(o, d)
        assert jnp.allclose(t1, t2, atol=1e-9)
        assert jnp.allclose(p1, p2, atol=1e-9)
        assert jnp.allclose(n1, n2, atol=1e-9)


# =============================================================================
# Zernike surfaces + composable Sum surfaces
# =============================================================================


class TestZernikeTerms:
    """The Noll Zernike basis itself."""

    def test_named_term_values(self):
        s6 = float(jnp.sqrt(6.0))
        s3 = float(jnp.sqrt(3.0))
        # Z1 (piston) = 1 everywhere
        assert jnp.allclose(
            zernike_terms(jnp.array([0.3, -0.7]), jnp.array([-0.4, 0.1]))[..., 0], 1.0
        )
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

    def test_rms_normalized_and_orthogonal(self):
        """Noll terms have unit RMS and are mutually orthogonal over the unit disk."""
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        # Rejection-sample points uniformly in the unit disk.
        u = jax.random.uniform(k1, (200000,), minval=-1.0, maxval=1.0)
        v = jax.random.uniform(k2, (200000,), minval=-1.0, maxval=1.0)
        inside = (u * u + v * v) <= 1.0
        terms = jnp.where(inside[:, None], zernike_terms(u, v), 0.0)  # (M, 11)
        gram = (terms.T @ terms) / jnp.sum(inside)
        # Unit RMS -> diagonal ~ 1; orthogonal -> off-diagonal ~ 0.
        assert jnp.allclose(jnp.diag(gram), 1.0, atol=0.03)
        off = gram - jnp.diag(jnp.diag(gram))
        assert jnp.max(jnp.abs(off)) < 0.03

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
            dzdx, dzdy = fd_slope(lambda xx, yy: elem._sag_local(xx, yy), x, y)
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

    def test_decenter_shifts_sampling_and_rezeros(self):
        """A decentered Zernike samples the map off-centre but stays re-zeroed
        (sag at the decenter point is zero for any coefficients)."""
        coeffs = [0.0, 0.0, 0.0, 0.0, 1e-3, 2e-3]  # astigmatism only
        offset = (0.4, -0.3)
        zg = _zernike(coeffs, r_norm=1.0, offset=offset)
        centered = _zernike(coeffs, r_norm=1.0, offset=(0.0, 0.0))
        # sag at decenter point is still zero
        assert float(zg.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
        # decentered sag(x,y) == centered intrinsic shifted and re-zeroed
        x, y = 0.2, 0.1
        ox, oy = offset
        expected = float(centered._index(0)._sag_intrinsic(x + ox, y + oy)) - float(
            centered._index(0)._sag_intrinsic(ox, oy)
        )
        assert float(zg.sag_at(0, x, y)) == pytest.approx(expected, rel=1e-6, abs=1e-12)

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


class TestSumSurfaceGroup:
    def test_additive(self):
        """Sum sag = component sags added (components with zero offset)."""
        asph = _asphere(0.05, conic=0.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4], r_norm=1.0)
        s = SumSurfaceGroup([asph, zg])
        for x, y in [(0.3, 0.2), (-0.5, 0.4), (0.8, -0.1)]:
            expected = float(asph.sag_at(0, x, y)) + float(zg.sag_at(0, x, y))
            assert float(s.sag_at(0, x, y)) == pytest.approx(expected, rel=1e-6, abs=1e-12)

    def test_normal_and_intersection_on_surface(self):
        asph = _asphere(0.05, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4, 3e-4], r_norm=1.2)
        s = SumSurfaceGroup([asph, zg])
        elem = s._index(0)
        origin = jnp.array([0.5, -0.3, 8.0])
        direction = jnp.array([0.03, 0.01, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, normal = s.intersect_at(0, origin, direction)
        hit = origin + t * direction
        assert np.allclose(np.asarray(point), np.asarray(hit), atol=1e-6)
        z_surf = s.sag_at(0, float(hit[0]), float(hit[1]))
        assert float(hit[2]) == pytest.approx(float(z_surf), abs=1e-6)
        # normal vs finite difference of the composite sag
        dzdx, dzdy = fd_slope(
            lambda xx, yy: elem._sag_local(xx, yy), float(point[0]), float(point[1])
        )
        expected = np.array([-dzdx, -dzdy, 1.0])
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(np.asarray(normal), expected, atol=1e-4)

    def test_validation(self):
        with pytest.raises(ValueError):
            SumSurfaceGroup([])
        # Mismatched element counts (N=1 asphere vs N=2 Zernike).
        zg_n2 = ZernikeSurfaceGroup(coeffs=jnp.zeros((2, 4)), r_norm=jnp.ones(2))
        with pytest.raises(ValueError):
            SumSurfaceGroup([_asphere(0.1), zg_n2])


class TestRenderPipeline:
    """Both render entry points accept the new surfaces (first-class)."""

    def test_transform_to_world_zernike_and_sum(self):
        asph = _asphere(0.08, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4], r_norm=0.5)
        for surface in (zg, SumSurfaceGroup([asph, zg])):
            group = mirror_group_with_surface(surface)
            points, normals, _ = group.transform_to_world()
            assert points.shape == (1, 64, 3)
            assert jnp.all(jnp.isfinite(points))
            assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)

    def test_intersect_at_vmapped_sum(self):
        asph = _asphere(0.08, conic=-1.0)
        zg = _zernike([0.0, 0.0, 0.0, 1e-3, 8e-4, -6e-4], r_norm=0.5)
        group = mirror_group_with_surface(SumSurfaceGroup([asph, zg]))
        n_rays = 32
        key = jax.random.key(7)
        xy = jax.random.uniform(key, (n_rays, 2), minval=-0.3, maxval=0.3)
        origins = jnp.concatenate([xy, jnp.full((n_rays, 1), 6.0)], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        ts, pts, norms = jax.vmap(lambda o, d: group.surface.intersect_at(0, o, d))(
            origins, directions
        )
        assert jnp.all(jnp.isfinite(ts))
        # Each hit lies on the surface.
        z_surf = jax.vmap(lambda p: group.surface.sag_at(0, p[0], p[1]))(pts)
        assert jnp.allclose(pts[:, 2], z_surf, atol=1e-6)


# =============================================================================
# Freeform (bicubic-interpolated) surfaces
# =============================================================================


def _grid_from_fn(fn, n=21, half=0.5):
    xs = jnp.linspace(-half, half, n)
    ys = jnp.linspace(-half, half, n)
    X, Y = jnp.meshgrid(xs, ys)  # X[j,i]=xs[i], Y[j,i]=ys[j]
    return fn(X, Y), xs, ys


def _freeform(fn, n=21, half=0.5, offset=(0.0, 0.0)):
    grid, xs, ys = _grid_from_fn(fn, n, half)
    surf = FreeformSurfaceGroup.from_extent(grid[None], half, half, offsets=jnp.asarray([offset]))
    return surf, xs, ys


class TestBicubicKernel:
    def test_node_exact(self):
        grid = jnp.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
        )
        for j in range(4):
            for i in range(4):
                assert float(bicubic_interp(grid, float(i), float(j))) == pytest.approx(
                    float(grid[j, i])
                )

    def test_clamps_out_of_range(self):
        grid = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        # far outside -> clamped to the corner value
        assert float(bicubic_interp(grid, -5.0, -5.0)) == pytest.approx(0.0)
        assert float(bicubic_interp(grid, 10.0, 10.0)) == pytest.approx(3.0)


class TestFreeformSurface:
    def test_interpolation_node_exact_and_accurate(self):
        def fn(x, y):
            return 0.02 * x**2 - 0.01 * y**2 + 0.003 * jnp.sin(3 * x)

        surf, xs, ys = _freeform(fn, n=41, half=0.5)
        elem = surf._index(0)
        # exact at nodes
        for i in (0, 20, 40):
            for j in (0, 13, 40):
                assert float(elem._sag_intrinsic(xs[i], ys[j])) == pytest.approx(
                    float(fn(xs[i], ys[j])), abs=1e-9
                )
        # accurate between nodes
        for x, y in [(0.137, -0.21), (-0.33, 0.08), (0.4, 0.4)]:
            assert float(elem._sag_intrinsic(x, y)) == pytest.approx(float(fn(x, y)), abs=1e-5)

    def test_normal_matches_finite_difference(self):
        def fn(x, y):
            return 0.03 * x**2 - 0.02 * y**2 + 0.01 * x * y + 0.004 * jnp.sin(4 * x)

        surf, _, _ = _freeform(fn, n=41)
        elem = surf._index(0)
        for x, y in [(0.2, 0.1), (-0.3, 0.25), (0.15, -0.4)]:
            _, normal = elem.compute_sag_and_normal_at(x, y)
            dzdx, dzdy = fd_slope(lambda xx, yy: elem._sag_local(xx, yy), x, y)
            expected = np.array([-dzdx, -dzdy, 1.0])
            expected /= np.linalg.norm(expected)
            assert np.allclose(np.asarray(normal), expected, atol=1e-4)
            assert float(jnp.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-10)

    def test_flat_grid_is_flat(self):
        surf = FreeformSurfaceGroup.from_extent(jnp.full((1, 9, 9), 0.7), 0.5, 0.5)
        elem = surf._index(0)
        for x, y in [(0.0, 0.0), (0.2, -0.3), (0.4, 0.1)]:
            assert float(elem._sag_local(x, y)) == pytest.approx(0.0, abs=1e-6)
            _, normal = elem.compute_sag_and_normal_at(x, y)
            # float32: the interpolated slope of a constant grid is eps-level noise
            assert np.allclose(np.asarray(normal), [0.0, 0.0, 1.0], atol=1e-6)

    def test_standalone_intersection_on_surface(self):
        def fn(x, y):
            return 0.05 * (x**2 + y**2)

        surf, _, _ = _freeform(fn, n=41)
        origin = jnp.array([0.2, -0.1, 5.0])
        direction = jnp.array([0.01, -0.02, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, _ = surf.intersect_at(0, origin, direction)
        hit = origin + t * direction
        assert np.allclose(np.asarray(point), np.asarray(hit), atol=1e-6)
        z_surf = surf.sag_at(0, float(hit[0]), float(hit[1]))
        assert float(hit[2]) == pytest.approx(float(z_surf), abs=1e-6)

    def test_decenter_shifts_sampling_and_rezeros(self):
        def fn(x, y):
            return 0.5 + 0.02 * x**2 + 0.01 * y**2  # non-zero at origin

        offset = (0.1, -0.15)
        surf, _, _ = _freeform(fn, offset=offset)
        centered, _, _ = _freeform(fn, offset=(0.0, 0.0))
        # sag is re-zeroed: zero at the decenter point even though fn is not.
        assert float(surf.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
        x, y = 0.1, 0.05
        ox, oy = offset
        expected = float(centered._index(0)._sag_intrinsic(x + ox, y + oy)) - float(
            centered._index(0)._sag_intrinsic(ox, oy)
        )
        assert float(surf.sag_at(0, x, y)) == pytest.approx(expected, abs=1e-6)

    def test_validation(self):
        with pytest.raises(ValueError):
            FreeformSurfaceGroup.from_extent(jnp.zeros((5, 5)), 0.5, 0.5)  # not 3D
        with pytest.raises(ValueError):
            FreeformSurfaceGroup(  # 1xW grid too small
                grid_z=jnp.zeros((1, 1, 4)),
                x0=0.0,
                y0=0.0,
                dx=1.0,
                dy=1.0,
            )


class TestFreeformInSum:
    def test_sum_intersection_and_normal(self):
        """A freeform figure error added to an asphere: sum is additive, and the
        composite intersection lands on the surface with a matching normal."""
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.06]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
        )
        free, _, _ = _freeform(lambda x, y: 0.001 * (x**2 - y**2), n=41)
        s = SumSurfaceGroup([asph, free])  # asphere first -> conic initial guess
        # additivity
        for x, y in [(0.2, 0.1), (-0.3, 0.25)]:
            expected = float(asph.sag_at(0, x, y)) + float(free.sag_at(0, x, y))
            assert float(s.sag_at(0, x, y)) == pytest.approx(expected, abs=1e-9)
        # intersection + normal
        origin = jnp.array([0.25, -0.15, 8.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, point, normal = s.intersect_at(0, origin, direction)
        z_surf = s.sag_at(0, float(point[0]), float(point[1]))
        assert float(point[2]) == pytest.approx(float(z_surf), abs=1e-6)
        elem = s._index(0)
        dzdx, dzdy = fd_slope(
            lambda xx, yy: elem._sag_local(xx, yy), float(point[0]), float(point[1])
        )
        expected = np.array([-dzdx, -dzdy, 1.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(np.asarray(normal), expected, atol=1e-4)


class TestFreeformSerializationGuard:
    """Freeform grids are not YAML-serializable; saving must fail loudly."""

    def test_save_raises(self):
        from iactrace import Telescope
        from iactrace.io import telescope_to_dict

        free, _, _ = _freeform(lambda x, y: 0.002 * (x**2 - y**2), n=9)
        tel = Telescope(mirror_groups=[mirror_group_with_surface(free, radius=0.4)], name="f")
        with pytest.raises(ValueError, match="FreeformSurfaceGroup"):
            telescope_to_dict(tel)


class TestFreeformRenderPipeline:
    def test_transform_to_world(self):
        free, _, _ = _freeform(lambda x, y: 0.003 * (x**2 - y**2), n=31, half=0.5)
        group = mirror_group_with_surface(free, radius=0.4)
        points, normals, _ = group.transform_to_world()
        assert points.shape == (1, 64, 3)
        assert jnp.all(jnp.isfinite(points))
        assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)
