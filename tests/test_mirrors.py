import jax
import jax.numpy as jnp
import pytest

from iactrace import MCIntegrator
from iactrace.telescope.mirrors import (
    AsphericDiskMirrorGroup,
    AsphericPolygonMirrorGroup,
    _point_in_convex_polygon,
    _polygon_area,
)
from iactrace.utils.sampling import sample_annulus, sample_disk


class TestSampleAnnulus:
    """Test the sample_annulus function for uniform annular sampling."""

    def test_samples_within_bounds(self, random_key):
        """All samples should be within the annular region."""
        inner_r, outer_r = 0.3, 1.0
        n_samples = 10000

        samples = sample_annulus(random_key, inner_r, outer_r, (n_samples,))
        radii = jnp.sqrt(samples[:, 0]**2 + samples[:, 1]**2)

        assert jnp.all(radii >= inner_r - 1e-10), "Samples inside inner radius"
        assert jnp.all(radii <= outer_r + 1e-10), "Samples outside outer radius"

    def test_uniform_distribution_in_area(self, random_key):
        """Samples should be uniformly distributed by area (not by radius)."""
        inner_r, outer_r = 0.5, 1.0
        n_samples = 50000

        samples = sample_annulus(random_key, inner_r, outer_r, (n_samples,))
        radii = jnp.sqrt(samples[:, 0]**2 + samples[:, 1]**2)

        # For uniform area distribution, P(r < R) = (R² - inner²) / (outer² - inner²)
        # Split annulus into inner and outer halves by area
        mid_r_sq = (inner_r**2 + outer_r**2) / 2
        mid_r = jnp.sqrt(mid_r_sq)

        inner_half_count = jnp.sum(radii < mid_r)

        # Should be approximately 50-50 split
        ratio = inner_half_count / n_samples
        assert 0.48 < ratio < 0.52, f"Area distribution not uniform: {ratio:.3f}"

    def test_angular_uniformity(self, random_key):
        """Samples should be uniformly distributed in angle."""
        inner_r, outer_r = 0.3, 1.0
        n_samples = 10000

        samples = sample_annulus(random_key, inner_r, outer_r, (n_samples,))
        angles = jnp.arctan2(samples[:, 1], samples[:, 0])

        # Check samples in each quadrant (should be ~25% each)
        q1 = jnp.sum((angles >= 0) & (angles < jnp.pi/2)) / n_samples
        q2 = jnp.sum((angles >= jnp.pi/2) & (angles < jnp.pi)) / n_samples
        q3 = jnp.sum((angles >= -jnp.pi) & (angles < -jnp.pi/2)) / n_samples
        q4 = jnp.sum((angles >= -jnp.pi/2) & (angles < 0)) / n_samples

        for q, name in [(q1, "Q1"), (q2, "Q2"), (q3, "Q3"), (q4, "Q4")]:
            assert 0.23 < q < 0.27, f"Angular distribution not uniform in {name}: {q:.3f}"

    def test_zero_inner_radius_matches_disk(self, random_key):
        """With inner_radius=0, sample_annulus should match sample_disk distribution."""
        outer_r = 1.0
        n_samples = 10000

        # Same key for both
        key1, key2 = jax.random.split(random_key)

        annulus_samples = sample_annulus(key1, 0.0, outer_r, (n_samples,))
        disk_samples = sample_disk(key1, (n_samples,)) * outer_r

        # Both should have same statistical properties
        annulus_radii = jnp.sqrt(annulus_samples[:, 0]**2 + annulus_samples[:, 1]**2)
        disk_radii = jnp.sqrt(disk_samples[:, 0]**2 + disk_samples[:, 1]**2)

        # Mean radius for uniform disk: 2/3 * R
        expected_mean = 2/3 * outer_r
        assert jnp.isclose(jnp.mean(annulus_radii), expected_mean, rtol=0.05)
        assert jnp.isclose(jnp.mean(disk_radii), expected_mean, rtol=0.05)


class TestAnnularAperture:
    """Test check_aperture for mirrors with center holes."""

    @pytest.fixture
    def annular_mirror_group(self):
        """Create a mirror group with a center hole."""
        return AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 1)),
            radii=jnp.array([1.0]),           # outer radius
            inner_radii=jnp.array([0.3]),     # center hole
        )

    def test_point_in_annulus_accepted(self, annular_mirror_group):
        """Points within the annular region should be accepted."""
        # Point at r=0.5, between inner (0.3) and outer (1.0)
        x, y = 0.5, 0.0
        assert annular_mirror_group.check_aperture(x, y, 0)

        # Point at r=0.7
        x, y = 0.5, 0.5
        assert annular_mirror_group.check_aperture(x, y, 0)

    def test_point_in_hole_rejected(self, annular_mirror_group):
        """Points inside the center hole should be rejected."""
        # Point at r=0.2, inside hole (inner_r=0.3)
        x, y = 0.2, 0.0
        assert not annular_mirror_group.check_aperture(x, y, 0)

        # Point at origin
        x, y = 0.0, 0.0
        assert not annular_mirror_group.check_aperture(x, y, 0)

        # Point at r=0.1
        x, y = 0.07, 0.07
        assert not annular_mirror_group.check_aperture(x, y, 0)

    def test_point_outside_rejected(self, annular_mirror_group):
        """Points outside the outer radius should be rejected."""
        # Point at r=1.5, outside outer radius (1.0)
        x, y = 1.5, 0.0
        assert not annular_mirror_group.check_aperture(x, y, 0)

    def test_boundary_inner(self, annular_mirror_group):
        """Points exactly on inner boundary should be accepted."""
        inner_r = 0.3
        x, y = inner_r, 0.0
        assert annular_mirror_group.check_aperture(x, y, 0)

    def test_boundary_outer(self, annular_mirror_group):
        """Points exactly on outer boundary should be accepted."""
        outer_r = 1.0
        x, y = outer_r, 0.0
        assert annular_mirror_group.check_aperture(x, y, 0)

    def test_vectorized_check(self, annular_mirror_group):
        """check_aperture should work with arrays of points."""
        x = jnp.array([0.0, 0.2, 0.5, 0.9, 1.5])
        y = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = annular_mirror_group.check_aperture(x, y, 0)
        expected = jnp.array([False, False, True, True, False])

        assert jnp.array_equal(result, expected)


class TestMirrorGroupWithHole:
    """Test AsphericDiskMirrorGroup with inner_radii parameter."""

    def test_inner_radii_defaults_to_zero(self):
        """inner_radii should default to zeros (solid disk)."""
        group = AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.1, 0.1]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 1)),
            radii=jnp.array([1.0, 0.5]),
        )

        assert jnp.allclose(group.inner_radii, jnp.array([0.0, 0.0]))

    def test_mixed_solid_and_annular(self):
        """Group can have mix of solid and annular mirrors."""
        group = AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.1, 0.1]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 1)),
            radii=jnp.array([1.0, 0.5]),
            inner_radii=jnp.array([0.2, 0.0]),  # First has hole, second is solid
        )

        # First mirror: point at origin rejected (in hole)
        assert not group.check_aperture(0.0, 0.0, 0)
        # First mirror: point at r=0.5 accepted
        assert group.check_aperture(0.5, 0.0, 0)

        # Second mirror: point at origin accepted (solid disk)
        assert group.check_aperture(0.0, 0.0, 1)

    def test_sampling_params_include_inner_radii(self):
        """get_sampling_params should include inner_radii."""
        group = AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 1)),
            radii=jnp.array([1.0]),
            inner_radii=jnp.array([0.3]),
        )

        params = group.get_sampling_params()

        assert 'inner_radii' in params
        assert jnp.allclose(params['inner_radii'], jnp.array([0.3]))
        assert params['type'] == 'disk'

    def test_area_calculation_annular(self, random_key):
        """Area should be π*(outer² - inner²) for annular aperture."""
        inner_r, outer_r = 0.3, 1.0
        expected_area = jnp.pi * (outer_r**2 - inner_r**2)

        group = AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.0]),  # Flat mirror for simple area check
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 1)),
            radii=jnp.array([outer_r]),
            inner_radii=jnp.array([inner_r]),
        )

        integrator = MCIntegrator(n_samples=1000)
        group = integrator.sample_group(group, random_key)

        _, _, weights = group.transform_to_world()

        # For flat mirror with normal along z, weights = 1/area * n_samples
        # So area = n_samples / weight (for single sample)
        computed_area = 1000 / weights[0, 0, 0]

        assert jnp.isclose(computed_area, expected_area, rtol=0.01)


class TestPolygonArea:
    """Test polygon area calculation."""

    def test_square_area(self):
        """Square with side 2 has area 4."""
        vertices = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        area = _polygon_area(vertices)
        assert jnp.isclose(area, 4.0)

    def test_regular_hexagon_area(self):
        """Regular hexagon area = 3*sqrt(3)/2 * s^2."""
        s = 1.0  # circumradius
        angles = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        vertices = jnp.stack([jnp.cos(angles) * s, jnp.sin(angles) * s], axis=1)
        area = _polygon_area(vertices)
        expected = 3 * jnp.sqrt(3) / 2 * s**2
        assert jnp.isclose(area, expected, rtol=1e-6)

    def test_triangle_area(self):
        """Right triangle with legs 2, 2 has area 2."""
        vertices = jnp.array([[0, 0], [2, 0], [0, 2]])
        area = _polygon_area(vertices)
        assert jnp.isclose(area, 2.0)


class TestPointInPolygon:
    """Test point-in-polygon checking."""

    def test_point_inside_square(self):
        """Points inside square are detected."""
        vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        assert _point_in_convex_polygon(0.0, 0.0, vertices, 4)
        assert _point_in_convex_polygon(0.5, 0.5, vertices, 4)
        assert _point_in_convex_polygon(-0.5, -0.5, vertices, 4)

    def test_point_outside_square(self):
        """Points outside square are rejected."""
        vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        assert not _point_in_convex_polygon(2.0, 0.0, vertices, 4)
        assert not _point_in_convex_polygon(0.0, 2.0, vertices, 4)
        assert not _point_in_convex_polygon(1.5, 1.5, vertices, 4)

    def test_point_on_boundary(self):
        """Points on boundary are included (>= check)."""
        vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        assert _point_in_convex_polygon(1.0, 0.0, vertices, 4)
        assert _point_in_convex_polygon(0.0, 1.0, vertices, 4)

    def test_vectorized_check(self):
        """Point check works with arrays."""
        vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        x = jnp.array([0.0, 0.5, 2.0, -0.5])
        y = jnp.array([0.0, 0.5, 0.0, 2.0])
        result = _point_in_convex_polygon(x, y, vertices, 4)
        expected = jnp.array([True, True, False, False])
        assert jnp.array_equal(result, expected)


class TestAsphericPolygonMirrorGroup:
    """Test polygon aperture mirror group."""

    @pytest.fixture
    def hexagon_mirror(self):
        """Create a single hexagonal mirror."""
        s = 0.5
        angles = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        vertices = jnp.stack([jnp.cos(angles) * s, jnp.sin(angles) * s], axis=1)

        return AsphericPolygonMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 1)),
            vertices_list=vertices[None, :, :],
        )

    def test_basic_creation(self, hexagon_mirror):
        """Polygon mirror group can be created."""
        assert len(hexagon_mirror) == 1
        assert hexagon_mirror.n_vertices == 6

    def test_check_aperture(self, hexagon_mirror):
        """Aperture checking works for polygon."""
        # Center point should be inside
        assert hexagon_mirror.check_aperture(0.0, 0.0, 0)
        # Point inside but off-center
        assert hexagon_mirror.check_aperture(0.1, 0.1, 0)
        # Point outside
        assert not hexagon_mirror.check_aperture(1.0, 0.0, 0)

    def test_from_config_roundtrip(self):
        """from_config creates equivalent group."""
        templates = {
            "hex_surface": {
                "surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []}
            }
        }

        configs = [
            {
                "template": "hex_surface",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "aperture": {
                    "type": "polygon",
                    "vertices": [[0.0, 0.5], [0.43, 0.25], [0.43, -0.25],
                                 [0.0, -0.5], [-0.43, -0.25], [-0.43, 0.25]],
                },
            }
        ]

        group = AsphericPolygonMirrorGroup.from_config(configs, templates)

        assert len(group) == 1
        assert group.n_vertices == 6
        assert float(group.curvatures[0]) == pytest.approx(0.05)


class TestMultipleMirrorsInGroup:
    """Test groups with multiple mirrors."""

    def test_disk_group_different_parameters(self):
        """Disk group with per-mirror surface parameters."""
        group = AsphericDiskMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.05, 0.1]),  # Different curvatures
            conics=jnp.array([-1.0, 0.0]),  # Different conics
            aspherics=jnp.zeros((2, 1)),
            radii=jnp.array([0.5, 0.3]),  # Different radii
        )

        assert len(group) == 2
        assert float(group.curvatures[0]) == pytest.approx(0.05)
        assert float(group.curvatures[1]) == pytest.approx(0.1)
        assert float(group.radii[0]) == pytest.approx(0.5)
        assert float(group.radii[1]) == pytest.approx(0.3)

    def test_polygon_group_multiple_mirrors(self):
        """Polygon group with multiple mirrors."""
        # Two square mirrors
        vertices = jnp.array([
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]],
        ])

        group = AsphericPolygonMirrorGroup(
            positions=jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.05, 0.1]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 1)),
            vertices_list=vertices,
        )

        assert len(group) == 2
        # Check aperture for each mirror
        assert group.check_aperture(0.0, 0.0, 0)
        assert group.check_aperture(0.4, 0.0, 0)
        assert not group.check_aperture(0.4, 0.0, 1)  # Outside smaller mirror
