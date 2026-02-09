import jax
import jax.numpy as jnp
import pytest

from iactrace import MCIntegrator, Telescope
from iactrace.telescope.mirrors import AsphericDiskMirrorGroup


class TestRoughnessInTracing:
    """Test that mirror roughness affects ray tracing results."""

    @pytest.fixture
    def telescope(self):
        """Load a test telescope."""
        return Telescope.from_yaml(
            'configs/HESS/CT3.yaml',
            MCIntegrator(16),
            key=jax.random.key(42)
        )

    @pytest.fixture
    def test_rays(self):
        """Create test rays for tracing."""
        n_rays = 500
        key = jax.random.key(123)
        key1, key2 = jax.random.split(key)
        r = 5.0 * jnp.sqrt(jax.random.uniform(key1, (n_rays,)))
        theta = jax.random.uniform(key2, (n_rays,)) * 2 * jnp.pi
        origins = jnp.stack([
            r * jnp.cos(theta),
            r * jnp.sin(theta),
            jnp.ones(n_rays) * 100.0
        ], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        values = jnp.ones(n_rays)
        return origins, directions, values

    def test_perturbation_key_initialized(self, telescope):
        """Test that perturbation_key is properly initialized."""
        group = telescope.mirror_groups[0]
        assert hasattr(group, 'perturbation_key')
        # Key should be a JAX array (not the default key(0))
        assert group.perturbation_key is not None

    def test_roughness_increases_psf_spread(self, telescope, test_rays):
        """Test that applying roughness increases the PSF spread."""
        origins, directions, values = test_rays

        # Trace without roughness
        pts_clean, _, vals_clean = telescope.trace(
            origins, directions, values, sensor_idx=0, debug=True
        )

        # Apply significant roughness (60 arcsec)
        tel_rough = telescope.apply_roughness(60.0)
        pts_rough, _, vals_rough = tel_rough.trace(
            origins, directions, values, sensor_idx=0, debug=True
        )

        # Compare PSF spread
        hit_mask_clean = vals_clean > 0
        hit_mask_rough = vals_rough > 0

        if jnp.sum(hit_mask_clean) > 10 and jnp.sum(hit_mask_rough) > 10:
            std_clean = jnp.std(pts_clean[hit_mask_clean, 0])
            std_rough = jnp.std(pts_rough[hit_mask_rough, 0])

            # Roughness should increase spread
            assert std_rough > std_clean, \
                f"Roughness should increase PSF spread: {std_rough} <= {std_clean}"

    def test_zero_roughness_no_change(self, telescope, test_rays):
        """Test that zero roughness doesn't change results."""
        origins, directions, values = test_rays

        # Trace without roughness
        pts_clean, _, vals_clean = telescope.trace(
            origins, directions, values, sensor_idx=0, debug=True
        )

        # Apply zero roughness
        tel_zero = telescope.apply_roughness(0.0)
        pts_zero, _, vals_zero = tel_zero.trace(
            origins, directions, values, sensor_idx=0, debug=True
        )

        # Results should be identical
        assert jnp.allclose(pts_clean, pts_zero), "Zero roughness should not change results"
        assert jnp.allclose(vals_clean, vals_zero), "Zero roughness should not change values"

    def test_roughness_is_deterministic(self, telescope, test_rays):
        """Test that roughness produces deterministic results."""
        origins, directions, values = test_rays

        tel_rough = telescope.apply_roughness(30.0)

        # Trace twice
        pts1, _, vals1 = tel_rough.trace(origins, directions, values, sensor_idx=0, debug=True)
        pts2, _, vals2 = tel_rough.trace(origins, directions, values, sensor_idx=0, debug=True)

        # Results should be identical
        assert jnp.allclose(pts1, pts2), "Roughness should be deterministic"
        assert jnp.allclose(vals1, vals2), "Roughness values should be deterministic"

    def test_different_keys_different_results(self, test_rays):
        """Test that different random keys produce different perturbations."""
        origins, directions, values = test_rays

        # Create two telescopes with different keys
        tel1 = Telescope.from_yaml(
            'configs/HESS/CT3.yaml',
            MCIntegrator(16),
            key=jax.random.key(1)
        ).apply_roughness(30.0)

        tel2 = Telescope.from_yaml(
            'configs/HESS/CT3.yaml',
            MCIntegrator(16),
            key=jax.random.key(2)
        ).apply_roughness(30.0)

        pts1, _, vals1 = tel1.trace(origins, directions, values, sensor_idx=0, debug=True)
        pts2, _, vals2 = tel2.trace(origins, directions, values, sensor_idx=0, debug=True)

        # Results should be different (not exactly equal)
        assert not jnp.allclose(pts1, pts2), \
            "Different keys should produce different perturbations"


class TestMirrorGroupPerturbationKey:
    """Test perturbation_key handling in MirrorGroup classes."""

    def test_disk_mirror_group_has_key(self):
        """Test that AsphericDiskMirrorGroup initializes perturbation_key."""
        n_mirrors = 3
        group = AsphericDiskMirrorGroup(
            positions=jnp.zeros((n_mirrors, 3)),
            rotations=jnp.zeros((n_mirrors, 3)),
            curvatures=jnp.ones(n_mirrors) * 0.01,
            conics=jnp.zeros(n_mirrors),
            aspherics=jnp.zeros((n_mirrors, 1)),
            radii=jnp.ones(n_mirrors) * 0.5,
        )

        assert hasattr(group, 'perturbation_key')
        assert group.perturbation_key is not None

    def test_perturbation_scale_default_zero(self):
        """Test that perturbation_scale defaults to zero."""
        n_mirrors = 3
        group = AsphericDiskMirrorGroup(
            positions=jnp.zeros((n_mirrors, 3)),
            rotations=jnp.zeros((n_mirrors, 3)),
            curvatures=jnp.ones(n_mirrors) * 0.01,
            conics=jnp.zeros(n_mirrors),
            aspherics=jnp.zeros((n_mirrors, 1)),
            radii=jnp.ones(n_mirrors) * 0.5,
        )

        assert jnp.all(group.perturbation_scale == 0), \
            "Default perturbation_scale should be zero"
