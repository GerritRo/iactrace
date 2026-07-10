import jax
import jax.numpy as jnp

from iactrace.core.bsdf import GaussianBSDF, _apply_perturbation
from iactrace.core.intersections import intersect_conic, newton_raphson_intersect
from iactrace.core.surfaces import compute_sag_and_normal, sag, sag_raw


class TestSurfaceSag:
    """Test surface sag calculations for known cases."""

    def test_flat_surface_has_zero_sag(self):
        """Flat surface (zero curvature) has zero sag everywhere."""
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)]:
            z = sag_raw(x, y, curvature=0.0, conic=0.0, aspheric=jnp.array([]))
            assert jnp.allclose(z, 0.0, atol=1e-12)

    def test_spherical_sag_symmetry(self):
        """Spherical surface sag depends only on r^2 = x^2 + y^2."""
        c = 0.05
        r = 2.0

        z1 = sag_raw(r, 0.0, c, 0.0, jnp.array([]))
        z2 = sag_raw(0.0, r, c, 0.0, jnp.array([]))
        z3 = sag_raw(r / jnp.sqrt(2), r / jnp.sqrt(2), c, 0.0, jnp.array([]))
        z4 = sag_raw(-r, 0.0, c, 0.0, jnp.array([]))

        assert jnp.allclose(z1, z2, atol=1e-12)
        assert jnp.allclose(z1, z3, atol=1e-12)
        assert jnp.allclose(z1, z4, atol=1e-12)

    def test_spherical_sag_formula(self):
        """Verify spherical sag matches analytical formula: z = c*r^2 / (1 + sqrt(1 - c^2*r^2))."""
        c = 0.02
        r = 3.0

        z_computed = sag_raw(r, 0.0, c, 0.0, jnp.array([]))

        z_expected = c * r**2 / (1 + jnp.sqrt(1 - c * c * r**2))

        assert jnp.allclose(z_computed, z_expected, atol=1e-12)

    def test_paraboloid_sag_formula(self):
        """Verify parabolic sag: z = c*r^2 / 2 for paraboloid (k=-1)."""
        c = 0.02
        r = 3.0

        z_computed = sag_raw(r, 0.0, c, conic=-1.0, aspheric=jnp.array([]))
        z_expected = c * r * r / 2.0

        assert jnp.allclose(z_computed, z_expected, atol=1e-12)

    def test_aspheric_powers_are_consecutive_even(self):
        """aspheric[i] must multiply r^(2*i+4): r^4, r^6, r^8, r^10, ..."""
        r = 0.7
        coeffs = jnp.array([1.1, -2.3, 3.7, -4.2])  # r^4, r^6, r^8, r^10
        z_computed = sag_raw(r, 0.0, curvature=0.0, conic=0.0, aspheric=coeffs)
        z_expected = coeffs[0] * r**4 + coeffs[1] * r**6 + coeffs[2] * r**8 + coeffs[3] * r**10
        assert jnp.allclose(z_computed, z_expected, atol=1e-12)

    def test_offset_sag_equals_zero_at_offset_point(self):
        """sag() with offset returns 0 at the offset point itself."""
        offset = jnp.array([2.0, 3.0])
        c = 0.05

        z = sag(0.0, 0.0, offset, c, 0.0, jnp.array([]))
        assert jnp.allclose(z, 0.0, atol=1e-12)


class TestSurfaceNormals:
    """Test that computed surface normals are correct."""

    def test_flat_surface_normal_is_z(self):
        """Flat surface has normal pointing in z direction everywhere."""
        offset = jnp.array([0.0, 0.0])

        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)]:
            _, normal = compute_sag_and_normal(x, y, offset, 0.0, 0.0, jnp.array([]))
            expected = jnp.array([0.0, 0.0, 1.0])
            assert jnp.allclose(normal, expected, atol=1e-8)

    def test_normal_is_unit_length(self):
        """Surface normal should always be unit length."""
        offset = jnp.array([0.0, 0.0])

        for c in [0.0, 0.01, 0.1]:
            for x, y in [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)]:
                _, normal = compute_sag_and_normal(x, y, offset, c, 0.0, jnp.array([]))
                assert jnp.allclose(jnp.linalg.norm(normal), 1.0, atol=1e-10)

    def test_spherical_normal_points_to_center(self):
        """For a spherical surface, normal at (x,y,z) points toward center of curvature."""
        c = 0.05  # curvature = 1/R, so R = 20
        R = 1.0 / c
        offset = jnp.array([0.0, 0.0])

        x, y = 2.0, 0.0
        point, normal = compute_sag_and_normal(x, y, offset, c, 0.0, jnp.array([]))

        # Vector from point to center of curvature
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

    def test_perturbed_normals_unit_length(self):
        """Perturbed normals remain unit vectors."""
        key = jax.random.key(42)
        normals = jnp.array([[0.0, 0.0, 1.0], [0.1, 0.2, 0.97], [-0.3, 0.1, 0.95]])
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)

        bsdf = GaussianBSDF(scale=jnp.array([0.01, 0.01, 0.01]))
        element_idx = jnp.array([0, 1, 2])
        perturbed = bsdf.perturb_normals(normals, key, element_idx)

        norms = jnp.linalg.norm(perturbed, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)

    def test_perturbation_is_small_for_small_scale(self):
        """Small perturbation scale produces small angular deviations."""
        key = jax.random.key(123)
        normal = jnp.array([[0.0, 0.0, 1.0]])

        bsdf = GaussianBSDF(scale=jnp.array([1e-4]))
        element_idx = jnp.array([0])
        perturbed = bsdf.perturb_normals(normal, key, element_idx)

        dot = jnp.sum(normal * perturbed)
        assert dot > 0.9999


class TestConicIntersection:
    """Test ray-conic surface intersection for different surface types."""

    def _surface_residual(self, point, c, k):
        """Check if point lies on conic surface: c*(x^2+y^2) + (1+k)*c*z^2 - 2z = 0."""
        x, y, z = point[0], point[1], point[2]
        return c * (x**2 + y**2) + (1 + k) * c * z**2 - 2 * z

    def test_flat_surface(self):
        """Flat surface (c=0) is plane at z=0."""
        t = intersect_conic(jnp.array([0.0, 0.0, 5.0]), jnp.array([0.0, 0.0, -1.0]), 0.0, 0.0)
        assert jnp.allclose(t, 5.0)

    def test_paraboloid_surface(self):
        """Paraboloid (k=-1) intersection lies on surface."""
        c = 0.05
        origin = jnp.array([3.0, 2.0, 15.0])
        direction = jnp.array([-0.1, -0.05, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t = intersect_conic(origin, direction, c, -1.0)
        hit = origin + t * direction
        assert jnp.allclose(self._surface_residual(hit, c, -1.0), 0.0, atol=1e-8)

    def test_hyperboloid_surface(self):
        """Hyperboloid (k < -1) intersection lies on surface."""
        c, k = 0.1, -2.0
        origin = jnp.array([0.5, 0.5, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t = intersect_conic(origin, direction, c, k)
        hit = origin + t * direction
        assert jnp.allclose(self._surface_residual(hit, c, k), 0.0, atol=1e-8)


class TestNewtonRaphsonIntersect:
    """Test Newton-Raphson ray-surface intersection for z = sag(x, y)."""

    def _verify_hit(self, sag_fn, ray_origin, ray_direction, t, hit_xy, valid, tol=1e-6):
        """Verify the returned intersection is correct."""
        if not valid:
            return
        hit_point = ray_origin + t * ray_direction
        assert jnp.allclose(hit_xy[0], hit_point[0], atol=tol)
        assert jnp.allclose(hit_xy[1], hit_point[1], atol=tol)
        z_surface = sag_fn(hit_xy[0], hit_xy[1])
        assert jnp.allclose(hit_point[2], z_surface, atol=tol)

    def test_flat_plane(self):
        """Flat plane z=0 intersection."""

        def sag_fn(x, y):
            return 0.0

        origin = jnp.array([3.0, 4.0, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        assert jnp.allclose(t, 10.0)
        assert jnp.allclose(hit_xy, jnp.array([3.0, 4.0]))

    def test_paraboloid(self):
        """Paraboloid z = c*(x^2 + y^2) intersection."""
        c = 0.1

        def sag_fn(x, y):
            return c * (x**2 + y**2)

        origin = jnp.array([2.0, 0.0, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        self._verify_hit(sag_fn, origin, direction, t, hit_xy, valid)
        assert jnp.allclose(t, 9.6, atol=1e-6)

    def test_spherical_sag(self):
        """Spherical sag z = c*(x^2 + y^2) / (1 + sqrt(1 - c^2*(x^2 + y^2)))."""
        c = 0.05

        def sag_fn(x, y):
            return c * (x**2 + y**2) / (1 + jnp.sqrt(1 - c**2 * (x**2 + y**2)))

        origin = jnp.array([2.0, 1.0, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        self._verify_hit(sag_fn, origin, direction, t, hit_xy, valid)

    def test_aspheric_surface(self):
        """Aspheric surface with higher-order terms."""
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
        self._verify_hit(sag_fn, origin, direction, t, hit_xy, valid)

    def test_ray_parallel_misses(self):
        """Ray parallel to flat surface misses."""

        def sag_fn(x, y):
            return 0.0

        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([1.0, 0.0, 0.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert not valid or jnp.isinf(t)

    def test_ray_pointing_away_misses(self):
        """Ray pointing away from surface misses."""

        def sag_fn(x, y):
            return 0.0

        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert not valid or jnp.isinf(t)

    def test_ray_behind_surface_misses(self):
        """Ray starting behind surface going away misses."""

        def sag_fn(x, y):
            return 0.0

        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert not valid
        assert jnp.isinf(t)

    def test_large_distances(self):
        """Works at large distances."""

        def sag_fn(x, y):
            return 0.0

        origin = jnp.array([0.0, 0.0, 1e6])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        assert jnp.allclose(t, 1e6, rtol=1e-6)

    def test_high_curvature(self):
        """Works with high curvature."""
        c = 1.0

        def sag_fn(x, y):
            return c * (x**2 + y**2)

        origin = jnp.array([0.5, 0.5, 10.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, hit_xy, valid = newton_raphson_intersect(sag_fn, origin, direction)
        assert valid
        self._verify_hit(sag_fn, origin, direction, t, hit_xy, valid)


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
