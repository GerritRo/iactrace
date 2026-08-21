import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Telescope
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import RefractInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.refractive_index import ConstantIndex
from iactrace.core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from iactrace.telescope import operations as ops

from ._helpers import make_disk_mirror_group, mirror_group_with_surface


@pytest.fixture
def simple_telescope(random_key):
    """Create a minimal telescope with known properties for testing."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    curvatures = jnp.array([0.1, 0.2])  # Known curvatures
    conics = jnp.array([-1.0, -1.0])
    aspherics = jnp.zeros((2, 2))
    radii = jnp.array([0.5, 0.5])

    mirror_group = make_disk_mirror_group(
        positions,
        rotations,
        curvatures,
        conics,
        aspherics,
        radii,
        optical_stage=0,
        n_samples=64,
    )

    return Telescope(mirror_groups=[mirror_group], obstruction_groups=None, name="test_telescope")


class TestMathematicalCorrectness:
    """Verify mathematical transformations are correct."""

    def test_roughness_stored_in_arcseconds(self, simple_telescope):
        """Roughness should be stored in arcseconds on the BSDF."""
        roughness_arcsec = 3600.0  # 1 degree = 3600 arcseconds
        modified_tel = ops.apply_roughness(simple_telescope, 0, roughness_arcsec)
        actual_scale = modified_tel.mirror_groups[0].bsdf.scale[0]
        assert jnp.isclose(actual_scale, roughness_arcsec, rtol=1e-10)

    def test_focal_length_to_curvature_conversion(self, simple_telescope):
        """Focal length converts via c = 1/(2f); infinite f gives a flat mirror."""
        focal_lengths = jnp.array([5.0, jnp.inf])
        modified_tel = ops.set_focal_lengths(simple_telescope, 0, focal_lengths)
        curvatures = modified_tel.mirror_groups[0].surface.curvatures
        assert jnp.isclose(curvatures[0], 0.1)  # 1/(2*5)
        assert curvatures[1] == 0.0  # infinite focal length -> flat

    def test_scale_and_offset_curvatures(self, simple_telescope):
        """Scaling multiplies and offsetting adds the mirror curvatures."""
        original = simple_telescope.mirror_groups[0].surface.curvatures.copy()

        scaled = ops.scale_curvatures(simple_telescope, 0, 2.5)
        assert jnp.allclose(scaled.mirror_groups[0].surface.curvatures, original * 2.5)

        offset = ops.offset_curvatures(simple_telescope, 0, 0.05)
        assert jnp.allclose(offset.mirror_groups[0].surface.curvatures, original + 0.05)


class TestRandomPerturbations:
    """Verify statistical properties of random perturbations."""

    @pytest.mark.slow
    def test_misalignment_has_correct_statistics(self, simple_telescope, random_key):
        """Misalignment perturbations should have correct mean and std."""
        n_mirrors = 1000
        positions = jnp.zeros((n_mirrors, 3))
        rotations = jnp.zeros((n_mirrors, 3))
        curvatures = jnp.full(n_mirrors, 0.1)
        conics = jnp.full(n_mirrors, -1.0)
        aspherics = jnp.zeros((n_mirrors, 2))
        radii = jnp.full(n_mirrors, 0.5)

        large_group = make_disk_mirror_group(
            positions,
            rotations,
            curvatures,
            conics,
            aspherics,
            radii,
            optical_stage=0,
            n_samples=4,
        )

        tel = Telescope(
            mirror_groups=[large_group], obstruction_groups=None, name="large_telescope"
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

        modified_tel = ops.apply_displacement(simple_telescope, 0, sigma_z=1.0, key=random_key)

        assert jnp.allclose(modified_tel.mirror_groups[0].positions[:, :2], original_xy)
        assert not jnp.allclose(
            modified_tel.mirror_groups[0].positions[:, 2],
            simple_telescope.mirror_groups[0].positions[:, 2],
        )

    def test_conic_error_preserves_mean(self, simple_telescope, random_key):
        """Conic error with zero sigma should preserve original values."""
        original_conics = simple_telescope.mirror_groups[0].surface.conics.copy()

        modified_tel = ops.apply_conic_error(simple_telescope, 0, sigma=0.0, key=random_key)

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


class TestGetInfo:
    """Test the get_info utility function."""

    def test_get_info_counts_and_bounding_box(self, simple_telescope):
        """get_info reports mirror counts, stage layout and the bounding box."""
        info = ops.get_info(simple_telescope)

        assert info["n_mirror_elements"] == 2
        assert info["n_lens_elements"] == 0
        assert info["n_stages"] == 1
        assert info["stages"] == [
            {"stage": 0, "kind": "mirror", "n_elements": 2, "aperture": "disk"}
        ]
        assert info["name"] == "test_telescope"
        # Mirrors at (0,0,0) and (1,0,0)
        assert jnp.allclose(info["bbox_min"], jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(info["bbox_max"], jnp.array([1.0, 0.0, 0.0]))


def _make_lens_telescope():
    """Telescope with one mirror group at stage 0 and one lens group at stage 1."""
    n = 2
    mirror = make_disk_mirror_group(
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
            index=ConstantIndex(jnp.full(n, 1.5)),
            transmittance_curve=None,
            transmittance=jnp.full(n, 0.9),
        ),
        sample_key=jax.random.key(1),
        optical_stage=1,
    )
    return Telescope(mirror_groups=[mirror], lens_groups=[lens], name="lens_test")


class TestLensOperations:
    """Lens stages accept the same generic ops as mirror stages, plus a few
    lens-specific ones (refractive index, transmittance, refractive focal formula)."""

    def test_generic_ops_apply_to_lens_stage(self):
        """set_positions / set_rotations / apply_misalignment / apply_roughness
        all work on a lens stage."""
        tel = _make_lens_telescope()

        new_pos = jnp.array([[1.0, 0.0, 4.0], [-1.0, 0.0, 4.0]])
        tel = tel.set_positions(stage=1, positions=new_pos)
        assert jnp.allclose(tel.stage(1).positions, new_pos)

        new_rot = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        tel = tel.set_rotations(stage=1, rotations=new_rot)
        assert jnp.allclose(tel.stage(1).rotations, new_rot)

        before = tel.stage(1).rotations
        tel = tel.apply_misalignment(stage=1, sigma_h=10.0, sigma_v=10.0, key=jax.random.key(7))
        assert not jnp.allclose(before, tel.stage(1).rotations)

        tel = tel.apply_roughness(stage=1, sigma=20.0)
        assert jnp.allclose(tel.stage(1).bsdf.scale, jnp.full(2, 20.0))

    def test_scale_transmittance_clipped(self):
        tel = _make_lens_telescope()
        tel = tel.scale_transmittance(stage=1, factor=0.5)
        assert jnp.allclose(tel.stage(1).interaction_module.transmittance, 0.45)
        tel2 = _make_lens_telescope().scale_transmittance(stage=1, factor=10.0)
        assert jnp.allclose(tel2.stage(1).interaction_module.transmittance, 1.0)

    def test_set_refractive_index_and_element_counts(self):
        tel = _make_lens_telescope()
        assert tel.n_lens_elements == 2
        assert tel.n_mirror_elements == 2
        tel = tel.set_refractive_index(stage=1, index=1.6)
        assert jnp.allclose(tel.stage(1).interaction_module.index.reference(), 1.6)

    def test_set_refractive_index_accepts_dispersion_model(self):
        from iactrace.core.refractive_index import TabulatedIndex

        tel = _make_lens_telescope()  # 2-element lens at stage 1
        disp = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=2)
        tel = tel.set_refractive_index(stage=1, index=disp)
        assert isinstance(tel.stage(1).interaction_module.index, TabulatedIndex)

    def test_set_refractive_index_rejects_wrong_element_count(self):
        tel = _make_lens_telescope()  # 2-element lens at stage 1
        with pytest.raises(ValueError, match="elements"):
            tel.set_refractive_index(stage=1, index=ConstantIndex(jnp.array([1.5])))  # N=1 != 2

    def test_set_focal_length_on_lens_uses_refractive_formula(self):
        # index=1.5, n_outside=1 -> c = 1/((n-1)*f); f=4 -> c = 1/(0.5*4) = 0.5
        tel = _make_lens_telescope().set_focal_lengths(stage=1, focal_lengths=jnp.full(2, 4.0))
        assert jnp.allclose(tel.stage(1).surface.curvatures, 0.5)


class TestKindValidation:
    """Kind-specific operations reject the wrong kind."""

    def test_kind_specific_ops_reject_wrong_kind(self):
        """Reflectivity/thickness ops reject a lens stage; transmittance rejects
        a mirror stage."""
        tel = _make_lens_telescope()
        with pytest.raises(ValueError, match="stage 1 is lens"):
            tel.scale_reflectivity(stage=1, factor=0.9)
        with pytest.raises(ValueError, match="stage 1 is lens"):
            tel.set_thickness(stage=1, thickness=0.005)
        with pytest.raises(ValueError, match="stage 0 is mirror"):
            tel.scale_transmittance(stage=0, factor=0.9)

    def test_stage_raises_on_missing_index(self):
        tel = _make_lens_telescope()
        with pytest.raises(IndexError, match="no stage 5"):
            tel.stage(5)


# =============================================================================
# Zernike figure-error operations
# =============================================================================


@pytest.fixture
def asphere_telescope():
    n = 2
    surface = AsphericSurfaceGroup(
        curvatures=jnp.array([0.1, 0.2]),
        conics=jnp.array([-1.0, -1.0]),
        aspherics=jnp.zeros((n, 0)),
        offsets=jnp.zeros((n, 2)),
    )
    group = mirror_group_with_surface(surface, radius=jnp.array([0.5, 0.5]))
    return Telescope(mirror_groups=[group], name="t")


class TestApplyZernikeError:
    def test_wraps_bare_asphere_in_sum(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        # asphere component preserved unchanged
        asph = ops._asphere_of(surface)
        assert isinstance(asph, AsphericSurfaceGroup)
        assert np.allclose(np.asarray(asph.curvatures), [0.1, 0.2])
        # zernike component present
        zg = ops._zernike_of(surface)
        assert isinstance(zg, ZernikeSurfaceGroup)

    def test_coefficients_match_draw_and_r_norm_from_aperture(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        zg = ops._zernike_of(tel.stage(0).surface)
        # coefficients are exactly the sigma-scaled normal draw for this key
        expected = jax.random.normal(random_key, (2, sigmas.shape[0])) * sigmas[None, :]
        assert np.allclose(np.asarray(zg.coeffs), np.asarray(expected))
        # normalization radius is taken from each element's aperture radius (0.5)
        assert np.allclose(np.asarray(zg.r_norm), [0.5, 0.5])

    def test_accumulates(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        k1, k2 = jax.random.split(random_key)
        tel1 = ops.apply_zernike_error(asphere_telescope, 0, sigmas, k1)
        tel2 = ops.apply_zernike_error(tel1, 0, sigmas, k2)
        c1 = ops._zernike_of(tel1.stage(0).surface).coeffs
        c2 = ops._zernike_of(tel2.stage(0).surface).coeffs
        draw2 = jax.random.normal(k2, (2, sigmas.shape[0])) * sigmas[None, :]
        # second application adds its draw on top of the first
        assert np.allclose(np.asarray(c2), np.asarray(c1 + draw2))
        # still a single Zernike term (not two)
        comps = tel2.stage(0).surface.components
        assert sum(isinstance(c, ZernikeSurfaceGroup) for c in comps) == 1

    def test_too_many_modes_raises(self, asphere_telescope, random_key):
        with pytest.raises(ValueError):
            ops.apply_zernike_error(asphere_telescope, 0, jnp.zeros(12), random_key)


class TestNamedAberrations:
    def _columns(self, tel):
        zg = ops._zernike_of(tel.stage(0).surface)
        return np.asarray(zg.coeffs)

    @pytest.mark.parametrize(
        ("apply", "n_cols", "zero_upto"),
        [
            (ops.apply_astigmatism, 6, 4),  # only Z5, Z6 nonzero
            (ops.apply_coma, 8, 6),  # only Z7, Z8 nonzero
            (ops.apply_trefoil, 10, 8),  # only Z9, Z10 nonzero
        ],
    )
    def test_named_aberration_masks(self, asphere_telescope, random_key, apply, n_cols, zero_upto):
        tel = apply(asphere_telescope, 0, 1e-3, random_key)
        cols = self._columns(tel)
        assert cols.shape[1] == n_cols
        assert np.allclose(cols[:, :zero_upto], 0.0)  # lower-order modes untouched
        assert np.any(cols[:, zero_upto:n_cols] != 0.0)  # the named pair is set

    def test_telescope_method(self, asphere_telescope, random_key):
        tel = asphere_telescope.apply_astigmatism(0, 1e-3, random_key)
        assert isinstance(tel.stage(0).surface, SumSurfaceGroup)


class TestCapabilityDispatchThroughSum:
    """Prescription operations keep working after a Zernike term is added."""

    def test_set_curvatures_through_sum(self, asphere_telescope, random_key):
        tel = ops.apply_zernike_error(
            asphere_telescope, 0, jnp.array([0.0, 0.0, 0.0, 1e-3]), random_key
        )
        tel = ops.set_curvatures(tel, 0, jnp.array([0.3, 0.4]))
        asph = ops._asphere_of(tel.stage(0).surface)
        assert np.allclose(np.asarray(asph.curvatures), [0.3, 0.4])
        # set_conics dispatches through the Sum wrapper too
        tel = ops.set_conics(tel, 0, jnp.array([0.0, 0.0]))
        assert np.allclose(np.asarray(ops._asphere_of(tel.stage(0).surface).conics), [0.0, 0.0])
        # Zernike term untouched throughout
        assert ops._zernike_of(tel.stage(0).surface) is not None

    def test_focal_error_through_sum(self, asphere_telescope, random_key):
        k1, k2 = jax.random.split(random_key)
        tel = ops.apply_zernike_error(asphere_telescope, 0, jnp.array([0.0, 0.0, 0.0, 1e-3]), k1)
        c_before = ops._asphere_of(tel.stage(0).surface).curvatures
        tel = ops.apply_focal_error(tel, 0, 0.05, k2)
        c_after = ops._asphere_of(tel.stage(0).surface).curvatures
        assert not np.allclose(np.asarray(c_before), np.asarray(c_after))

    def test_no_asphere_raises(self, random_key):
        """A standalone-Zernike stage rejects aspheric prescription ops."""
        zg = ZernikeSurfaceGroup(coeffs=jnp.zeros((1, 4)), r_norm=jnp.ones(1))
        group = mirror_group_with_surface(zg, radius=jnp.array([0.5]))
        tel = Telescope(mirror_groups=[group], name="z")
        with pytest.raises(ValueError, match="no aspheric surface"):
            ops.set_curvatures(tel, 0, jnp.array([0.1]))


class TestEndToEndTrace:
    def test_zernike_error_perturbs_trace(self, asphere_telescope, random_key):
        n_rays = 64
        key = jax.random.key(3)
        xy = jax.random.uniform(key, (n_rays, 2), minval=-0.3, maxval=0.3)
        origins = jnp.concatenate([xy, jnp.full((n_rays, 1), 5.0)], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        values = jnp.ones(n_rays)

        rb_clean = asphere_telescope.trace(origins, directions, values).rays
        tel = asphere_telescope.apply_astigmatism(0, 5e-3, random_key)
        rb_pert = tel.trace(origins, directions, values).rays

        assert jnp.all(jnp.isfinite(rb_pert.directions))
        # the reflected directions should change under a real figure error
        assert not jnp.allclose(rb_clean.directions, rb_pert.directions)
