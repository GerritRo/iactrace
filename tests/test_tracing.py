import jax
import jax.numpy as jnp
import pytest

from iactrace import Camera, Telescope
from iactrace.camera.camera import intersect_sensor
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import AsphericSurfaceGroup


def _xy(camera, rb):
    """Return (x, y) of intersected sensor positions; for tests."""
    sensor_rays, _, _ = intersect_sensor(camera, rb)
    return sensor_rays.origins[:, 0], sensor_rays.origins[:, 1]


class TestRoughnessInTracing:
    """Test that mirror roughness affects ray tracing results."""

    @pytest.fixture
    def telescope_and_camera(self):
        """Load a test telescope and camera from the split config files."""
        telescope = Telescope.from_yaml(
            "configs/HESS/CT3.yaml",
            16,
            key=jax.random.key(42),
        )
        camera = Camera.from_yaml("configs/HESS/HESS1U.yaml")
        return telescope, camera

    @pytest.fixture
    def test_rays(self):
        """Create test rays for tracing."""
        n_rays = 500
        key = jax.random.key(123)
        key1, key2 = jax.random.split(key)
        r = 5.0 * jnp.sqrt(jax.random.uniform(key1, (n_rays,)))
        theta = jax.random.uniform(key2, (n_rays,)) * 2 * jnp.pi
        origins = jnp.stack(
            [r * jnp.cos(theta), r * jnp.sin(theta), jnp.ones(n_rays) * 100.0], axis=1
        )
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        values = jnp.ones(n_rays)
        return origins, directions, values

    def test_sample_key_initialized(self, telescope_and_camera):
        """Test that sample_key is properly initialized."""
        telescope, camera = telescope_and_camera
        group = telescope.mirror_groups[0]
        assert hasattr(group, "sample_key")
        assert group.sample_key is not None

    def test_roughness_increases_psf_spread(self, telescope_and_camera, test_rays):
        """Test that applying roughness increases the PSF spread."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        # Trace without roughness
        rb_clean = telescope.trace(origins, directions, values)
        x_clean, y_clean = _xy(camera, rb_clean)

        # Apply significant roughness (60 arcsec)
        tel_rough = telescope.apply_roughness(0, 60.0)
        rb_rough = tel_rough.trace(origins, directions, values)
        x_rough, y_rough = _xy(camera, rb_rough)

        # Compare PSF spread
        hit_mask_clean = rb_clean.values > 0
        hit_mask_rough = rb_rough.values > 0

        if jnp.sum(hit_mask_clean) > 10 and jnp.sum(hit_mask_rough) > 10:
            std_clean = jnp.std(x_clean[hit_mask_clean])
            std_rough = jnp.std(x_rough[hit_mask_rough])

            # Roughness should increase spread
            assert std_rough > std_clean, (
                f"Roughness should increase PSF spread: {std_rough} <= {std_clean}"
            )

    def test_zero_roughness_no_change(self, telescope_and_camera, test_rays):
        """Test that zero roughness doesn't change results."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        # Trace without roughness
        rb_clean = telescope.trace(origins, directions, values)
        x_clean, y_clean = _xy(camera, rb_clean)

        # Apply zero roughness
        tel_zero = telescope.apply_roughness(0, 0.0)
        rb_zero = tel_zero.trace(origins, directions, values)
        x_zero, y_zero = _xy(camera, rb_zero)

        # Results should be identical
        pts_clean = jnp.stack([x_clean, y_clean], axis=1)
        pts_zero = jnp.stack([x_zero, y_zero], axis=1)
        assert jnp.allclose(pts_clean, pts_zero), "Zero roughness should not change results"
        assert jnp.allclose(rb_clean.values, rb_zero.values), (
            "Zero roughness should not change values"
        )

    def test_roughness_is_deterministic(self, telescope_and_camera, test_rays):
        """Test that roughness produces deterministic results."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        tel_rough = telescope.apply_roughness(0, 30.0)

        # Trace twice
        rb1 = tel_rough.trace(origins, directions, values)
        rb2 = tel_rough.trace(origins, directions, values)
        x1, y1 = _xy(camera, rb1)
        x2, y2 = _xy(camera, rb2)

        # Results should be identical
        pts1 = jnp.stack([x1, y1], axis=1)
        pts2 = jnp.stack([x2, y2], axis=1)
        assert jnp.allclose(pts1, pts2), "Roughness should be deterministic"
        assert jnp.allclose(rb1.values, rb2.values), "Roughness values should be deterministic"

    def test_different_keys_different_results(self, test_rays):
        """Test that different random keys produce different perturbations."""
        origins, directions, values = test_rays

        # Create two telescopes with different keys; they share a camera.
        tel1 = Telescope.from_yaml(
            "configs/HESS/CT3.yaml",
            16,
            key=jax.random.key(1),
        )
        tel1 = tel1.apply_roughness(0, 30.0)
        cam1 = Camera.from_yaml("configs/HESS/HESS1U.yaml")

        tel2 = Telescope.from_yaml(
            "configs/HESS/CT3.yaml",
            16,
            key=jax.random.key(2),
        )
        tel2 = tel2.apply_roughness(0, 30.0)
        cam2 = Camera.from_yaml("configs/HESS/HESS1U.yaml")

        rb1 = tel1.trace(origins, directions, values)
        rb2 = tel2.trace(origins, directions, values)
        x1, y1 = _xy(cam1, rb1)
        x2, y2 = _xy(cam2, rb2)

        pts1 = jnp.stack([x1, y1], axis=1)
        pts2 = jnp.stack([x2, y2], axis=1)

        # Results should be different (not exactly equal)
        assert not jnp.allclose(pts1, pts2), "Different keys should produce different perturbations"


class TestMirrorGroupKeys:
    """Test sample_key and bsdf handling in OpticalElementGroup."""

    def test_disk_mirror_group_has_key(self):
        """Test that OpticalElementGroup initializes sample_key."""
        n_mirrors = 3
        surface = AsphericSurfaceGroup(
            curvatures=jnp.ones(n_mirrors) * 0.01,
            conics=jnp.zeros(n_mirrors),
            aspherics=jnp.zeros((n_mirrors, 1)),
            offsets=jnp.zeros((n_mirrors, 2)),
        )
        aperture = DiskAperture(
            radii=jnp.ones(n_mirrors) * 0.5,
            inner_radii=jnp.zeros(n_mirrors),
        )
        interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n_mirrors))
        group = OpticalElementGroup(
            positions=jnp.zeros((n_mirrors, 3)),
            rotations=jnp.zeros((n_mirrors, 3)),
            surface=surface,
            aperture=aperture,
            interaction_module=interaction,
            sample_key=jax.random.key(0),
            optical_stage=0,
            n_samples=100,
        )

        assert hasattr(group, "sample_key")
        assert group.sample_key is not None

    def test_bsdf_scale_default_zero(self):
        """Test that bsdf.scale defaults to zero when no bsdf is set."""
        n_mirrors = 3
        surface = AsphericSurfaceGroup(
            curvatures=jnp.ones(n_mirrors) * 0.01,
            conics=jnp.zeros(n_mirrors),
            aspherics=jnp.zeros((n_mirrors, 1)),
            offsets=jnp.zeros((n_mirrors, 2)),
        )
        aperture = DiskAperture(
            radii=jnp.ones(n_mirrors) * 0.5,
            inner_radii=jnp.zeros(n_mirrors),
        )
        interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n_mirrors))
        group = OpticalElementGroup(
            positions=jnp.zeros((n_mirrors, 3)),
            rotations=jnp.zeros((n_mirrors, 3)),
            surface=surface,
            aperture=aperture,
            interaction_module=interaction,
            sample_key=jax.random.key(0),
            optical_stage=0,
            n_samples=100,
        )

        # bsdf.scale defaults to zero (no roughness effect)
        assert jnp.all(group.bsdf.scale == 0), "Default bsdf.scale should be zero"
