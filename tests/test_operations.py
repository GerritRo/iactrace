import jax
import jax.numpy as jnp
import pytest

from iactrace import Telescope
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction, RefractInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import AsphericSurfaceGroup
from iactrace.telescope import operations as ops


def _make_disk_mirror_group(positions, rotations, curvatures, conics, aspherics,
                            radii, optical_stage=0, n_samples=100):
    """Build an OpticalElementGroup configured as a reflective disk mirror."""
    n = curvatures.shape[0]
    surface = AsphericSurfaceGroup(
        curvatures=curvatures,
        conics=conics,
        aspherics=aspherics,
        offsets=jnp.zeros((n, 2)),
    )
    aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=positions,
        rotations=rotations,
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=optical_stage,
        n_samples=n_samples,
    )


@pytest.fixture
def simple_telescope(random_key):
    """Create a minimal telescope with known properties for testing."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    curvatures = jnp.array([0.1, 0.2])  # Known curvatures
    conics = jnp.array([-1.0, -1.0])
    aspherics = jnp.zeros((2, 2))
    radii = jnp.array([0.5, 0.5])

    mirror_group = _make_disk_mirror_group(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0, n_samples=64,
    )

    return Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=None,
        name="test_telescope"
    )


class TestMathematicalCorrectness:
    """Verify mathematical transformations are correct."""

    def test_roughness_stored_in_arcseconds(self, simple_telescope):
        """Roughness should be stored in arcseconds on the BSDF."""
        roughness_arcsec = 3600.0  # 1 degree = 3600 arcseconds

        modified_tel = ops.apply_roughness(simple_telescope, 0, roughness_arcsec)

        actual_scale = modified_tel.mirror_groups[0].bsdf.scale[0]
        assert jnp.isclose(actual_scale, roughness_arcsec, rtol=1e-10)

    def test_focal_length_to_curvature_conversion(self, simple_telescope):
        """Focal length should convert correctly to curvature via c = 1/(2f)."""
        focal_lengths = jnp.array([5.0, 10.0])  # meters
        expected_curvatures = jnp.array([0.1, 0.05])  # 1/(2*f)

        modified_tel = ops.set_focal_lengths(simple_telescope, 0, focal_lengths)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].surface.curvatures,
            expected_curvatures,
            rtol=1e-10
        )

    def test_infinite_focal_length_gives_zero_curvature(self, simple_telescope):
        """Infinite focal length (flat mirror) should give zero curvature."""
        focal_lengths = jnp.array([jnp.inf, 10.0])

        modified_tel = ops.set_focal_lengths(simple_telescope, 0, focal_lengths)

        assert modified_tel.mirror_groups[0].surface.curvatures[0] == 0.0
        assert jnp.isclose(modified_tel.mirror_groups[0].surface.curvatures[1], 0.05)

    def test_scale_curvatures_multiplies_correctly(self, simple_telescope):
        """Scaling curvatures should multiply by scale factor."""
        original = simple_telescope.mirror_groups[0].surface.curvatures.copy()
        scale = 2.5

        modified_tel = ops.scale_curvatures(simple_telescope, 0, scale)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].surface.curvatures,
            original * scale
        )

    def test_offset_curvatures_adds_correctly(self, simple_telescope):
        """Offsetting curvatures should add the offset."""
        original = simple_telescope.mirror_groups[0].surface.curvatures.copy()
        offset = 0.05

        modified_tel = ops.offset_curvatures(simple_telescope, 0, offset)

        assert jnp.allclose(
            modified_tel.mirror_groups[0].surface.curvatures,
            original + offset
        )


class TestRandomPerturbations:
    """Verify statistical properties of random perturbations."""

    def test_misalignment_has_correct_statistics(self, simple_telescope, random_key):
        """Misalignment perturbations should have correct mean and std."""
        n_mirrors = 1000
        positions = jnp.zeros((n_mirrors, 3))
        rotations = jnp.zeros((n_mirrors, 3))
        curvatures = jnp.full(n_mirrors, 0.1)
        conics = jnp.full(n_mirrors, -1.0)
        aspherics = jnp.zeros((n_mirrors, 2))
        radii = jnp.full(n_mirrors, 0.5)

        large_group = _make_disk_mirror_group(
            positions, rotations, curvatures, conics, aspherics, radii,
            optical_stage=0, n_samples=4,
        )

        tel = Telescope(
            mirror_groups=[large_group],
            obstruction_groups=None,
            name="large_telescope"
        )

        sigma_h = 10.0  # arcseconds
        sigma_v = 20.0  # arcseconds
        key = jax.random.key(123)

        modified_tel = ops.apply_misalignment(tel, 0, sigma_h, sigma_v, key)

        delta_v = modified_tel.mirror_groups[0].rotations[:, 0]
        delta_h = modified_tel.mirror_groups[0].rotations[:, 1]

        expected_sigma_h_deg = sigma_h / 3600.0
        expected_sigma_v_deg = sigma_v / 3600.0

        assert jnp.abs(jnp.mean(delta_h)) < 3 * expected_sigma_h_deg / jnp.sqrt(n_mirrors)
        assert jnp.abs(jnp.mean(delta_v)) < 3 * expected_sigma_v_deg / jnp.sqrt(n_mirrors)

        assert jnp.isclose(jnp.std(delta_h), expected_sigma_h_deg, rtol=0.1)
        assert jnp.isclose(jnp.std(delta_v), expected_sigma_v_deg, rtol=0.1)

    def test_displacement_affects_only_z(self, simple_telescope, random_key):
        """Z-axis displacement should only affect z coordinates."""
        original_xy = simple_telescope.mirror_groups[0].positions[:, :2].copy()

        modified_tel = ops.apply_displacement(
            simple_telescope, 0, sigma_z=1.0, key=random_key
        )

        assert jnp.allclose(
            modified_tel.mirror_groups[0].positions[:, :2],
            original_xy
        )
        assert not jnp.allclose(
            modified_tel.mirror_groups[0].positions[:, 2],
            simple_telescope.mirror_groups[0].positions[:, 2]
        )

    def test_conic_error_preserves_mean(self, simple_telescope, random_key):
        """Conic error with zero sigma should preserve original values."""
        original_conics = simple_telescope.mirror_groups[0].surface.conics.copy()

        modified_tel = ops.apply_conic_error(
            simple_telescope, 0, sigma=0.0, key=random_key
        )

        assert jnp.allclose(modified_tel.mirror_groups[0].surface.conics, original_conics)


class TestFocalErrorModes:
    """Test absolute vs relative focal length errors."""

    def test_relative_focal_error_scales_with_focal_length(self, simple_telescope, random_key):
        """Relative error should produce larger absolute errors for larger focal lengths."""
        sigma_relative = 0.1

        n_trials = 100
        errors_mirror0 = []
        errors_mirror1 = []

        for i in range(n_trials):
            key = jax.random.key(i)
            modified = ops.apply_focal_error(
                simple_telescope, 0, sigma_relative, key, relative=True
            )
            c0 = modified.mirror_groups[0].surface.curvatures[0]
            c1 = modified.mirror_groups[0].surface.curvatures[1]
            f0 = 1.0 / (2.0 * c0) if c0 != 0 else jnp.inf
            f1 = 1.0 / (2.0 * c1) if c1 != 0 else jnp.inf

            f0_orig = 5.0
            f1_orig = 2.5

            errors_mirror0.append(jnp.abs(f0 - f0_orig))
            errors_mirror1.append(jnp.abs(f1 - f1_orig))

        mean_error_0 = jnp.mean(jnp.array(errors_mirror0))
        mean_error_1 = jnp.mean(jnp.array(errors_mirror1))

        assert jnp.isclose(mean_error_0 / mean_error_1, 2.0, rtol=0.3)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_remove_obstruction_from_empty_raises(self, simple_telescope):
        """Removing obstruction when none exist should raise IndexError."""
        with pytest.raises(IndexError):
            ops.remove_obstruction(simple_telescope, 0)


class TestOperationComposition:
    """Test that operations can be composed correctly."""

    def test_multiple_operations_chain(self, simple_telescope):
        """Multiple operations should compose correctly."""
        tel = simple_telescope

        tel = ops.scale_curvatures(tel, 0, 1.5)
        tel = ops.apply_roughness(tel, 0, 10.0)

        assert jnp.allclose(
            tel.mirror_groups[0].surface.curvatures,
            simple_telescope.mirror_groups[0].surface.curvatures * 1.5
        )
        assert jnp.allclose(
            tel.mirror_groups[0].bsdf.scale,
            jnp.full(2, 10.0)
        )

class TestGetInfo:
    """Test the get_info utility function."""

    def test_get_info_returns_correct_counts(self, simple_telescope):
        """get_info should return correct mirror counts."""
        info = ops.get_info(simple_telescope)

        assert info["n_mirror_elements"] == 2
        assert info["n_lens_elements"] == 0
        assert info["n_stages"] == 1
        assert info["stages"] == [
            {"stage": 0, "kind": "mirror", "n_elements": 2, "aperture": "disk"}
        ]
        assert info["name"] == "test_telescope"

    def test_get_info_computes_bounding_box(self, simple_telescope):
        """get_info should compute correct mirror bounding box."""
        info = ops.get_info(simple_telescope)

        # Mirrors at (0,0,0) and (1,0,0)
        assert jnp.allclose(info["bbox_min"], jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(info["bbox_max"], jnp.array([1.0, 0.0, 0.0]))


def _make_lens_telescope():
    """Telescope with one mirror group at stage 0 and one lens group at stage 1."""
    n = 2
    mirror = _make_disk_mirror_group(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        curvatures=jnp.full(n, 0.1),
        conics=jnp.full(n, -1.0),
        aspherics=jnp.zeros((n, 1)),
        radii=jnp.full(n, 0.5),
        optical_stage=0,
        n_samples=16,
    )

    surface = AsphericSurfaceGroup(
        curvatures=jnp.full(n, 0.05),
        conics=jnp.zeros(n),
        aspherics=jnp.zeros((n, 1)),
        offsets=jnp.zeros((n, 2)),
    )
    lens = OpticalElementGroup(
        positions=jnp.array([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]]),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=DiskAperture(radii=jnp.full(n, 0.3), inner_radii=jnp.zeros(n)),
        interaction_module=RefractInteraction(
            n_inside=jnp.full(n, 1.5),
            n_outside=1.0,
            transmittance=None,
            transmittance_scalar=jnp.full(n, 0.9),
        ),
        sample_key=jax.random.key(1),
        optical_stage=1,
    )
    return Telescope(mirror_groups=[mirror], lens_groups=[lens], name="lens_test")


class TestLensOperations:
    """Lens stages should accept the same generic ops as mirror stages."""

    def test_set_positions_on_lens_stage(self):
        tel = _make_lens_telescope()
        new_pos = jnp.array([[1.0, 0.0, 4.0], [-1.0, 0.0, 4.0]])
        tel = tel.set_positions(stage=1, positions=new_pos)
        assert jnp.allclose(tel.stage(1).positions, new_pos)

    def test_set_rotations_on_lens_stage(self):
        tel = _make_lens_telescope()
        new_rot = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        tel = tel.set_rotations(stage=1, rotations=new_rot)
        assert jnp.allclose(tel.stage(1).rotations, new_rot)

    def test_scale_transmittance_clipped(self):
        tel = _make_lens_telescope()
        tel = tel.scale_transmittance(stage=1, factor=0.5)
        assert jnp.allclose(tel.stage(1).interaction_module.transmittance_scalar, 0.45)
        tel2 = _make_lens_telescope().scale_transmittance(stage=1, factor=10.0)
        assert jnp.allclose(tel2.stage(1).interaction_module.transmittance_scalar, 1.0)

    def test_apply_misalignment_on_lens_stage(self):
        tel = _make_lens_telescope()
        original = tel.stage(1).rotations
        tel = tel.apply_misalignment(
            stage=1, sigma_h=10.0, sigma_v=10.0, key=jax.random.key(7),
        )
        assert not jnp.allclose(original, tel.stage(1).rotations)

    def test_apply_roughness_on_lens_stage(self):
        tel = _make_lens_telescope()
        tel = tel.apply_roughness(stage=1, sigma=20.0)
        assert jnp.allclose(tel.stage(1).bsdf.scale, jnp.full(2, 20.0))

    def test_set_refractive_index(self):
        tel = _make_lens_telescope()
        tel = tel.set_refractive_index(stage=1, n_inside=1.6)
        assert jnp.allclose(tel.stage(1).interaction_module.n_inside, 1.6)

    def test_n_lens_elements(self):
        assert _make_lens_telescope().n_lens_elements == 2

    def test_n_mirror_elements(self):
        assert _make_lens_telescope().n_mirror_elements == 2

    def test_set_focal_length_on_lens_uses_refractive_formula(self):
        # n_inside=1.5, n_outside=1 → c = 1/((n-1)*f)
        tel = _make_lens_telescope().set_focal_lengths(
            stage=1, focal_lengths=jnp.full(2, 4.0)
        )
        # expected curvature = 1 / (0.5 * 4) = 0.5
        assert jnp.allclose(tel.stage(1).surface.curvatures, 0.5)


class TestKindValidation:
    """Kind-specific operations reject the wrong kind."""

    def test_scale_reflectivity_rejects_lens(self):
        tel = _make_lens_telescope()
        with pytest.raises(ValueError, match="stage 1 is lens"):
            tel.scale_reflectivity(stage=1, factor=0.9)

    def test_scale_transmittance_rejects_mirror(self):
        tel = _make_lens_telescope()
        with pytest.raises(ValueError, match="stage 0 is mirror"):
            tel.scale_transmittance(stage=0, factor=0.9)

    def test_set_thickness_rejects_lens(self):
        tel = _make_lens_telescope()
        with pytest.raises(ValueError, match="stage 1 is lens"):
            tel.set_thickness(stage=1, thickness=0.005)

    def test_stage_raises_on_missing_index(self):
        tel = _make_lens_telescope()
        with pytest.raises(IndexError, match="no stage 5"):
            tel.stage(5)


class TestStageAccess:
    """Stage indexing and kind queries."""

    def test_stage_indices_sorted(self):
        tel = _make_lens_telescope()
        assert tel.stage_indices() == [0, 1]

    def test_stages_of_kind(self):
        tel = _make_lens_telescope()
        assert tel.stages_of_kind("mirror") == [0]
        assert tel.stages_of_kind("lens") == [1]
        assert tel.stages_of_kind("slab") == []

    def test_n_stages(self):
        assert _make_lens_telescope().n_stages == 2

    def test_stage_kind_property(self):
        tel = _make_lens_telescope()
        assert tel.stage(0).kind == "mirror"
        assert tel.stage(1).kind == "lens"
