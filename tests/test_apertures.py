import jax
import jax.numpy as jnp
import pytest

from iactrace.core.apertures import (
    DiskAperture,
    PolygonAperture,
    _point_in_convex_polygon,
    _polygon_area,
)
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.sampling import sample_annulus
from iactrace.core.surfaces import AsphericSurfaceGroup


@pytest.mark.slow
class TestSampleAnnulus:
    """Test the sample_annulus function for uniform annular sampling."""

    def test_samples_within_bounds(self, random_key):
        """All samples should be within the annular region."""
        inner_r, outer_r = 0.3, 1.0
        n_samples = 10000

        samples = sample_annulus(random_key, inner_r, outer_r, (n_samples,))
        radii = jnp.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)

        assert jnp.all(radii >= inner_r - 1e-10), "Samples inside inner radius"
        assert jnp.all(radii <= outer_r + 1e-10), "Samples outside outer radius"

    def test_uniform_distribution_in_area(self, random_key):
        """Samples should be uniformly distributed by area (not by radius)."""
        inner_r, outer_r = 0.5, 1.0
        n_samples = 50000

        samples = sample_annulus(random_key, inner_r, outer_r, (n_samples,))
        radii = jnp.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)

        # For uniform area distribution, P(r < R) = (R^2 - inner^2) / (outer^2 - inner^2)
        # Split annulus into inner and outer halves by area
        mid_r_sq = (inner_r**2 + outer_r**2) / 2
        mid_r = jnp.sqrt(mid_r_sq)

        inner_half_count = jnp.sum(radii < mid_r)

        # Should be approximately 50-50 split
        ratio = inner_half_count / n_samples
        assert 0.48 < ratio < 0.52, f"Area distribution not uniform: {ratio:.3f}"


def _disk_group(radii, inner_radii, curvatures=None, conics=None):
    """A disk-aperture mirror group for aperture-masking tests."""
    n = radii.shape[0]
    surface = AsphericSurfaceGroup(
        curvatures=jnp.full(n, 0.1) if curvatures is None else curvatures,
        conics=jnp.full(n, -1.0) if conics is None else conics,
        aspherics=jnp.zeros((n, 1)),
        offsets=jnp.zeros((n, 2)),
    )
    aperture = DiskAperture(radii=radii, inner_radii=inner_radii)
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=0,
        n_samples=100,
    )


class TestAnnularAperture:
    """check_aperture for mirrors with a center hole (inner_r=0.3, outer_r=1.0)."""

    @pytest.fixture
    def annular_mirror_group(self):
        return _disk_group(jnp.array([1.0]), jnp.array([0.3]))

    def test_hole_body_outside_and_boundaries(self, annular_mirror_group):
        """One array covers the hole, the annular body, and outside the rim; the
        inner and outer boundaries themselves are accepted."""
        x = jnp.array([0.0, 0.2, 0.5, 0.9, 1.5])
        y = jnp.zeros(5)
        result = annular_mirror_group.check_aperture(x, y, 0)
        assert jnp.array_equal(result, jnp.array([False, False, True, True, False]))
        assert annular_mirror_group.check_aperture(0.3, 0.0, 0)  # inner boundary
        assert annular_mirror_group.check_aperture(1.0, 0.0, 0)  # outer boundary


class TestMirrorGroupWithHole:
    """Per-element apertures and area weighting for DiskAperture with inner_radii."""

    def test_mixed_solid_and_annular(self):
        """A group can mix a holed mirror and a solid mirror; each masks per-element."""
        group = _disk_group(
            jnp.array([1.0, 0.5]),
            jnp.array([0.2, 0.0]),  # first holed, second solid
        )
        # First mirror: origin is inside its hole -> rejected; r=0.5 accepted.
        assert not group.check_aperture(0.0, 0.0, 0)
        assert group.check_aperture(0.5, 0.0, 0)
        # Second mirror: solid disk -> origin accepted.
        assert group.check_aperture(0.0, 0.0, 1)

    def test_area_calculation_annular(self):
        """transform_to_world weights encode area = pi*(outer^2 - inner^2)."""
        inner_r, outer_r = 0.3, 1.0
        expected_area = jnp.pi * (outer_r**2 - inner_r**2)
        group = _disk_group(
            jnp.array([outer_r]),
            jnp.array([inner_r]),
            curvatures=jnp.array([0.0]),  # flat -> weights = n_samples / area
            conics=jnp.array([0.0]),
        )

        _, _, weights = group.transform_to_world()
        computed_area = 100 / weights[0, 0, 0]
        assert jnp.isclose(computed_area, expected_area, rtol=0.01)


class TestPolygonHelpers:
    """Low-level polygon area and point-in-polygon primitives."""

    def test_polygon_area(self):
        """Area formula for square, right triangle, and regular hexagon."""
        square = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        assert jnp.isclose(_polygon_area(square), 4.0)

        triangle = jnp.array([[0, 0], [2, 0], [0, 2]])
        assert jnp.isclose(_polygon_area(triangle), 2.0)

        angles = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        hexagon = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        assert jnp.isclose(_polygon_area(hexagon), 3 * jnp.sqrt(3) / 2, rtol=1e-6)

    def test_point_in_convex_polygon(self):
        """Inside/outside a unit square (as an array), and boundary points are
        included (>= check)."""
        vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        x = jnp.array([0.0, 0.5, 2.0, -0.5])
        y = jnp.array([0.0, 0.5, 0.0, 2.0])
        assert jnp.array_equal(
            _point_in_convex_polygon(x, y, vertices, 4), jnp.array([True, True, False, False])
        )
        assert _point_in_convex_polygon(1.0, 0.0, vertices, 4)  # on the +x edge
        assert _point_in_convex_polygon(0.0, 1.0, vertices, 4)  # on the +y edge


def _polygon_group(vertices, n_vertices, curvatures, conics):
    """A polygon-aperture mirror group."""
    n = curvatures.shape[0]
    surface = AsphericSurfaceGroup(
        curvatures=curvatures,
        conics=conics,
        aspherics=jnp.zeros((n, 1)),
        offsets=jnp.zeros((n, 2)),
    )
    aperture = PolygonAperture(vertices=vertices, n_vertices=n_vertices)
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])[:n],
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=0,
        n_samples=100,
    )


class TestPolygonMirrorGroup:
    """Polygon-aperture mirror groups mask points against their polygon."""

    def test_single_hexagon_check_aperture(self):
        s = 0.5
        angles = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        vertices = jnp.stack([jnp.cos(angles) * s, jnp.sin(angles) * s], axis=1)
        group = _polygon_group(vertices[None, :, :], 6, jnp.array([0.1]), jnp.array([-1.0]))
        assert group.aperture.n_vertices == 6
        assert group.check_aperture(0.0, 0.0, 0)  # centre
        assert group.check_aperture(0.1, 0.1, 0)  # off-centre, inside
        assert not group.check_aperture(1.0, 0.0, 0)  # outside

    def test_per_mirror_polygon_masking(self):
        """Two square mirrors of different size mask independently."""
        vertices = jnp.array(
            [
                [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]],
            ]
        )
        group = _polygon_group(vertices, 4, jnp.array([0.05, 0.1]), jnp.array([-1.0, -1.0]))
        assert len(group) == 2
        assert group.check_aperture(0.4, 0.0, 0)  # inside larger mirror
        assert not group.check_aperture(0.4, 0.0, 1)  # outside smaller mirror
