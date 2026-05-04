import jax
import jax.numpy as jnp
import pytest

from iactrace import MCIntegrator, SquareSensorGroup, Telescope
from iactrace.telescope import operations as ops
from iactrace.telescope.mirrors import AsphericDiskMirrorGroup


@pytest.fixture
def simple_telescope(random_key):
    """Create a minimal telescope with known properties for testing."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    curvatures = jnp.array([0.1, 0.2])  # Known curvatures
    conics = jnp.array([-1.0, -1.0])
    aspherics = jnp.zeros((2, 2))
    radii = jnp.array([0.5, 0.5])

    mirror_group = AsphericDiskMirrorGroup(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0
    )

    integrator = MCIntegrator(n_samples=64)
    mirror_group = integrator.sample_group(mirror_group, random_key)

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 5.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100, height=100,
        bounds=(-0.05, 0.05, -0.05, 0.05),
    )

    return Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=None,
        sensors=[sensor],
        name="test_telescope"
    )


class TestMathematicalCorrectness:
    """Verify mathematical transformations are correct."""

    def test_roughness_conversion_to_radians(self, simple_telescope):
        """Roughness in arcseconds should convert correctly to radians."""
        roughness_arcsec = 3600.0  # 1 degree = 3600 arcseconds
        expected_rad = jnp.pi / 180.0  # 1 degree in radians

        modified_tel = ops.apply_roughness_to_group(simple_telescope, 0, roughness_arcsec)

        actual_scale = modified_tel.mirror_groups[0].perturbation_scale[0]
        assert jnp.isclose(actual_scale, expected_rad, rtol=1e-10)

    def test_focal_length_to_curvature_conversion(self, simple_telescope):
        """Focal length should convert correctly to curvature via c = 1/(2f)."""
        focal_lengths = jnp.array([5.0, 10.0])  # meters
        expected_curvatures = jnp.array([0.1, 0.05])  # 1/(2*f)

        modified_tel = ops.set_focal_lengths(simple_telescope, 0, focal_lengths)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].curvatures,
            expected_curvatures,
            rtol=1e-10
        )

    def test_infinite_focal_length_gives_zero_curvature(self, simple_telescope):
        """Infinite focal length (flat mirror) should give zero curvature."""
        focal_lengths = jnp.array([jnp.inf, 10.0])

        modified_tel = ops.set_focal_lengths(simple_telescope, 0, focal_lengths)

        assert modified_tel.mirror_groups[0].curvatures[0] == 0.0
        assert jnp.isclose(modified_tel.mirror_groups[0].curvatures[1], 0.05)

    def test_scale_curvatures_multiplies_correctly(self, simple_telescope):
        """Scaling curvatures should multiply by scale factor."""
        original = simple_telescope.mirror_groups[0].curvatures.copy()
        scale = 2.5

        modified_tel = ops.scale_mirror_curvatures(simple_telescope, 0, scale)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].curvatures,
            original * scale
        )

    def test_offset_curvatures_adds_correctly(self, simple_telescope):
        """Offsetting curvatures should add the offset."""
        original = simple_telescope.mirror_groups[0].curvatures.copy()
        offset = 0.05

        modified_tel = ops.offset_mirror_curvatures(simple_telescope, 0, offset)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].curvatures,
            original + offset
        )


class TestRandomPerturbations:
    """Verify statistical properties of random perturbations."""

    def test_misalignment_has_correct_statistics(self, simple_telescope, random_key):
        """Misalignment perturbations should have correct mean and std."""
        # Use many mirrors for statistical testing
        n_mirrors = 1000
        positions = jnp.zeros((n_mirrors, 3))
        rotations = jnp.zeros((n_mirrors, 3))
        curvatures = jnp.full(n_mirrors, 0.1)
        conics = jnp.full(n_mirrors, -1.0)
        aspherics = jnp.zeros((n_mirrors, 2))
        radii = jnp.full(n_mirrors, 0.5)

        large_group = AsphericDiskMirrorGroup(
            positions, rotations, curvatures, conics, aspherics, radii,
            optical_stage=0
        )
        integrator = MCIntegrator(n_samples=4)
        large_group = integrator.sample_group(large_group, random_key)

        tel = Telescope(
            mirror_groups=[large_group],
            obstruction_groups=None,
            sensors=simple_telescope.sensors,
            name="large_telescope"
        )

        sigma_h = 10.0  # arcseconds
        sigma_v = 20.0  # arcseconds
        key = jax.random.key(123)

        modified_tel = ops.apply_misalignment_to_group(tel, 0, sigma_h, sigma_v, key)

        # Get the perturbations (difference from original zero rotations)
        delta_v = modified_tel.mirror_groups[0].rotations[:, 0]  # degrees
        delta_h = modified_tel.mirror_groups[0].rotations[:, 1]  # degrees

        # Convert expected sigma from arcsec to degrees
        expected_sigma_h_deg = sigma_h / 3600.0
        expected_sigma_v_deg = sigma_v / 3600.0

        # Check mean is near zero (within 3 sigma / sqrt(n))
        assert jnp.abs(jnp.mean(delta_h)) < 3 * expected_sigma_h_deg / jnp.sqrt(n_mirrors)
        assert jnp.abs(jnp.mean(delta_v)) < 3 * expected_sigma_v_deg / jnp.sqrt(n_mirrors)

        # Check std is within 10% of expected
        assert jnp.isclose(jnp.std(delta_h), expected_sigma_h_deg, rtol=0.1)
        assert jnp.isclose(jnp.std(delta_v), expected_sigma_v_deg, rtol=0.1)

    def test_displacement_along_local_z_for_untilted_mirrors(
        self, simple_telescope, random_key
    ):
        """For mirrors with zero rotation, local z == global z, so displacement
        should only affect the world-frame z coordinate."""
        original_xy = simple_telescope.mirror_groups[0].positions[:, :2].copy()

        modified_tel = ops.apply_displacement_to_group(
            simple_telescope, 0, sigma_z=1.0, key=random_key
        )

        # X and Y unchanged (mirrors have zero rotation in the fixture)
        assert jnp.allclose(
            modified_tel.mirror_groups[0].positions[:, :2],
            original_xy
        )
        # Z changed
        assert not jnp.allclose(
            modified_tel.mirror_groups[0].positions[:, 2],
            simple_telescope.mirror_groups[0].positions[:, 2]
        
    def test_conic_error_preserves_mean(self, simple_telescope, random_key):
        """Conic error with zero sigma should preserve original values."""
        original_conics = simple_telescope.mirror_groups[0].conics.copy()

        modified_tel = ops.apply_conic_error_to_group(
            simple_telescope, 0, sigma=0.0, key=random_key
        )

        assert jnp.allclose(modified_tel.mirror_groups[0].conics, original_conics)


class TestFocalErrorModes:
    """Test absolute vs relative focal length errors."""

    def test_relative_focal_error_scales_with_focal_length(self, simple_telescope, random_key):
        """Relative error should produce larger absolute errors for larger focal lengths."""
        # Curvatures 0.1 and 0.2 give focal lengths 5.0 and 2.5
        sigma_relative = 0.1  # 10% error

        # Run many trials to get statistics
        n_trials = 100
        errors_mirror0 = []
        errors_mirror1 = []

        for i in range(n_trials):
            key = jax.random.key(i)
            modified = ops.apply_focal_error_to_group(
                simple_telescope, 0, sigma_relative, key, relative=True
            )
            # Convert back to focal length
            c0 = modified.mirror_groups[0].curvatures[0]
            c1 = modified.mirror_groups[0].curvatures[1]
            f0 = 1.0 / (2.0 * c0) if c0 != 0 else jnp.inf
            f1 = 1.0 / (2.0 * c1) if c1 != 0 else jnp.inf

            # Original focal lengths
            f0_orig = 5.0  # 1/(2*0.1)
            f1_orig = 2.5  # 1/(2*0.2)

            errors_mirror0.append(jnp.abs(f0 - f0_orig))
            errors_mirror1.append(jnp.abs(f1 - f1_orig))

        # Mirror 0 has larger focal length, so should have larger absolute errors
        mean_error_0 = jnp.mean(jnp.array(errors_mirror0))
        mean_error_1 = jnp.mean(jnp.array(errors_mirror1))

        # Ratio should be close to ratio of focal lengths (5.0/2.5 = 2.0)
        assert jnp.isclose(mean_error_0 / mean_error_1, 2.0, rtol=0.3)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_replace_sensor_invalid_index_raises(self, simple_telescope):
        """Replacing sensor at invalid index should raise IndexError."""
        new_sensor = simple_telescope.sensors[0]

        with pytest.raises(IndexError):
            ops.replace_sensor(simple_telescope, new_sensor, idx=99)

        with pytest.raises(IndexError):
            ops.replace_sensor(simple_telescope, new_sensor, idx=-1)

    def test_remove_sensor_invalid_index_raises(self, simple_telescope):
        """Removing sensor at invalid index should raise IndexError."""
        with pytest.raises(IndexError):
            ops.remove_sensor(simple_telescope, idx=99)

    def test_remove_obstruction_from_empty_raises(self, simple_telescope):
        """Removing obstruction when none exist should raise IndexError."""
        with pytest.raises(IndexError):
            ops.remove_obstruction(simple_telescope, 0)


class TestOperationComposition:
    """Test that operations can be composed correctly."""

    def test_multiple_operations_chain(self, simple_telescope):
        """Multiple operations should compose correctly."""
        tel = simple_telescope

        # Chain multiple operations
        tel = ops.focus(tel, 0.1)
        tel = ops.scale_mirror_curvatures(tel, 0, 1.5)
        tel = ops.apply_roughness_to_group(tel, 0, 10.0)

        # Verify all changes applied
        assert jnp.isclose(
            tel.sensors[0].positions[0, 2],
            simple_telescope.sensors[0].positions[0, 2] + 0.1
        )
        assert jnp.allclose(
            tel.mirror_groups[0].curvatures,
            simple_telescope.mirror_groups[0].curvatures * 1.5
        )
        expected_roughness_rad = 10.0 * jnp.pi / (180.0 * 3600.0)
        assert jnp.allclose(
            tel.mirror_groups[0].perturbation_scale,
            jnp.full(2, expected_roughness_rad)
        )

    def test_clone_creates_independent_copy(self, simple_telescope):
        """Cloned telescope should be independent of original."""
        cloned = ops.clone(simple_telescope)

        # Modify clone
        cloned = ops.focus(cloned, 1.0)

        # Original unchanged
        assert simple_telescope.sensors[0].positions[0, 2] != cloned.sensors[0].positions[0, 2]


class TestGetInfo:
    """Test the get_info utility function."""

    def test_get_info_returns_correct_counts(self, simple_telescope):
        """get_info should return correct mirror and sensor counts."""
        info = ops.get_info(simple_telescope)

        assert info["n_mirrors"] == 2
        assert info["n_sensor_groups"] == 1
        assert info["n_sensors_total"] == 1  # 1 sensor in the group
        assert info["n_mirror_groups"] == 1
        assert info["optical_stages"] == [0]
        assert info["name"] == "test_telescope"

    def test_get_info_computes_bounding_box(self, simple_telescope):
        """get_info should compute correct mirror bounding box."""
        info = ops.get_info(simple_telescope)

        # Mirrors at (0,0,0) and (1,0,0)
        assert jnp.allclose(info["bbox_min"], jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(info["bbox_max"], jnp.array([1.0, 0.0, 0.0]))
