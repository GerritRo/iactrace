import jax
import jax.numpy as jnp

from iactrace import MCIntegrator, SquareSensorGroup, Telescope
from iactrace.core.obstructions import CylinderGroup
from iactrace.core.render import render_response_matrix
from iactrace.telescope.mirrors import AsphericDiskMirrorGroup


def make_simple_telescope(curvature=1.0, n_samples=1024, key=None):
    """Create a minimal telescope for testing: single parabolic mirror + sensor."""
    if key is None:
        key = jax.random.key(0)

    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([curvature])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = AsphericDiskMirrorGroup(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0
    )

    integrator = MCIntegrator(n_samples=n_samples)
    key, subkey = jax.random.split(key)
    mirror_group = integrator.sample_group(mirror_group, subkey)

    focal_length = 1.0 / (2.0 * curvature) if curvature != 0 else 1000.0

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, focal_length]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )

    return Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=None,
        sensors=[sensor],
        name="test_telescope"
    )


def make_two_stage_telescope(n_samples=512, key=None):
    """Create a two-stage telescope to test multi-stage processing.

    Uses two mirrors in series: first focuses light, second redirects to sensor.
    This tests the optical_stage mechanism without requiring precise Cassegrain geometry.
    """
    if key is None:
        key = jax.random.key(0)

    integrator = MCIntegrator(n_samples=n_samples)

    # First mirror - flat, reflects to second mirror
    first = AsphericDiskMirrorGroup(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        rotations=jnp.array([[0.0, 0.0, 0.0]]),
        curvatures=jnp.array([0.0]),  # flat
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.1]),
        optical_stage=0
    )
    key, subkey = jax.random.split(key)
    first = integrator.sample_group(first, subkey)

    # Second mirror - flat, tilted to send light to sensor
    second = AsphericDiskMirrorGroup(
        positions=jnp.array([[0.0, 0.0, 0.5]]),
        rotations=jnp.array([[0.0, 45.0, 0.0]]),  # tilted 45 degrees
        curvatures=jnp.array([0.0]),
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.2]),  # larger to catch reflected rays
        optical_stage=1  # Second stage
    )
    key, subkey = jax.random.split(key)
    second = integrator.sample_group(second, subkey)

    # Sensor to the side (where 45-degree reflection sends light)
    sensor = SquareSensorGroup(
        positions=[[0.5, 0.0, 0.5]],
        rotations=[[0.0, 90.0, 0.0]],  # facing the second mirror
        width=50,
        height=50,
        bounds=(-0.2, 0.2, -0.2, 0.2),
    )

    return Telescope(
        mirror_groups=[first, second],
        obstruction_groups=None,
        sensors=[sensor],
        name="two_stage_telescope"
    )


def make_telescope_with_obstruction(n_samples=1024, key=None):
    """Create a telescope with a central obstruction blocking incoming rays.

    The obstruction is placed above the mirror to block incoming parallel rays
    before they hit the mirror. This simulates a secondary mirror support structure.
    """
    if key is None:
        key = jax.random.key(0)

    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([1.0])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = AsphericDiskMirrorGroup(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0
    )

    integrator = MCIntegrator(n_samples=n_samples)
    key, subkey = jax.random.split(key)
    mirror_group = integrator.sample_group(mirror_group, subkey)

    # Central obstruction - cylinder blocking incoming rays
    # For parallel source with direction (0,0,-1), light comes from +z.
    # Shadow check traces back from mirror in -(-dir) = +z direction.
    # Obstruction must be at +z (between source and mirror, or mirror and sensor).
    # We place it between z=0.1 and z=0.4 (after reflection, before sensor at z=0.5).
    obstruction = CylinderGroup(
        p1=[[0.0, 0.0, 0.1]],
        p2=[[0.0, 0.0, 0.4]],
        r=[0.03],  # 30% of mirror radius
    )

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.5]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )

    return Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=[obstruction],
        sensors=[sensor],
        name="obstructed_telescope"
    )


class TestBasicRendering:
    """Test basic rendering functionality."""

    def test_render_returns_correct_shape(self):
        """Render output has expected shape."""
        tel = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = tel.render(sources, values, source_type='point')

        assert image.shape == (1, 100, 100)

    def test_render_nonzero_for_on_axis_source(self):
        """On-axis point source produces nonzero signal."""
        tel = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = tel.render(sources, values, source_type='point')

        assert jnp.sum(image) > 0

    def test_parallel_rays_converge_at_center(self):
        """Parallel rays (on-axis) should focus at image center within precision."""
        tel = make_simple_telescope()

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        points, sensor_idx, vals = tel.render(sources, values, source_type='parallel', debug=True)

        assert jnp.std(points[:, 0]) < 1e-8
        assert jnp.std(points[:, 1]) < 1e-8


class TestEnergyConservation:
    """Test radiometric properties of ray tracing."""

    def test_output_scales_with_input_intensity(self):
        """Output flux scales linearly with input intensity."""
        tel = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])

        values1 = jnp.array([1.0])
        values2 = jnp.array([3.0])

        image1 = tel.render(sources, values1, source_type='point')
        image2 = tel.render(sources, values2, source_type='point')

        ratio = jnp.sum(image2) / jnp.sum(image1)
        assert jnp.isclose(ratio, 3.0, rtol=0.01)

    def test_output_scales_with_mirror_area(self):
        """Output flux is proportional to mirror collecting area."""
        tel = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = tel.render(sources, values, source_type='point')
        total_flux = jnp.sum(image)

        mirror_area = jnp.pi * 0.1**2

        assert total_flux > 0.5 * mirror_area * values[0]
        assert total_flux < 1.5 * mirror_area * values[0]

    def test_reflectivity_scales_output(self):
        """Scaling mirror reflectivity scales output proportionally."""
        tel = make_simple_telescope()

        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image_full = tel.render(sources, values, source_type='point')
        flux_full = jnp.sum(image_full)

        tel_scaled = tel.scale_mirror_weights(0, 3)
        image_scaled = tel_scaled.render(sources, values, source_type='point')
        flux_scaled = jnp.sum(image_scaled)

        assert jnp.isclose(flux_scaled / flux_full, 3.0, rtol=0.01)


class TestResponseMatrix:
    """Test response matrix computation."""

    def test_response_matrix_shape(self):
        """Response matrix has correct shape."""
        tel = make_simple_telescope()
        sources = jnp.array([
            [0.0, 0.0, 1e6],
            [10.0, 0.0, 1e6],
            [0.0, 10.0, 1e6],
        ])
        values = jnp.array([1.0, 1.0, 1.0])

        R = render_response_matrix(tel, sources, values, 'point')

        assert R.shape == (3, 100 * 100)

    def test_response_matrix_row_sums_scale_with_area(self):
        """Row sums scale with mirror area (radiometric normalization)."""
        tel = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        R = render_response_matrix(tel, sources, values, 'point')
        row_sum = jnp.sum(R[0])

        mirror_area = jnp.pi * 0.1**2

        assert row_sum > 0.5 * mirror_area * values[0]
        assert row_sum < 1.5 * mirror_area * values[0]


class TestMultiStageRendering:
    """Test multi-stage optical systems with multiple mirrors."""

    def test_two_stage_has_correct_stages(self):
        """Two-stage telescope has mirrors in different optical stages."""
        tel = make_two_stage_telescope()

        stages = [g.optical_stage for g in tel.mirror_groups]
        assert stages == [0, 1]
        assert len(tel.mirror_groups) == 2

    def test_single_stage_produces_output(self):
        """Single-stage telescope works correctly (baseline)."""
        tel = make_simple_telescope()

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        image = tel.render(sources, values, source_type='parallel')
        assert jnp.sum(image) > 0


class TestObstructionEffects:
    """Test that obstructions block light correctly."""

    def test_obstruction_reduces_flux(self):
        """Central obstruction reduces total collected flux."""
        key = jax.random.key(42)

        tel_clear = make_simple_telescope(n_samples=2048, key=key)
        tel_obstructed = make_telescope_with_obstruction(n_samples=2048, key=key)

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        image_clear = tel_clear.render(sources, values, source_type='parallel')
        image_obstructed = tel_obstructed.render(sources, values, source_type='parallel')

        flux_clear = jnp.sum(image_clear)
        flux_obstructed = jnp.sum(image_obstructed)

        # Obstruction should reduce flux
        assert flux_obstructed < flux_clear

        # With cylinder r=0.03, mirror r=0.1, the obstruction blocks
        # rays in the central region, reducing total flux
        assert flux_obstructed > 0  # Some light still gets through

    def test_obstruction_group_is_attached(self):
        """Telescope with obstruction has the obstruction group attached."""
        tel = make_telescope_with_obstruction()

        assert tel.obstruction_groups is not None
        assert len(tel.obstruction_groups) == 1
        assert len(tel.obstruction_groups[0]) == 1  # One cylinder
