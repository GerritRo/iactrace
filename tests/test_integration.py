import jax
import jax.numpy as jnp
import pytest

from iactrace import Camera, Telescope
from iactrace.camera.camera import intersect_sensor
from iactrace.core.obstructions import SphereGroup

from ._helpers import (
    make_simple_telescope,
    make_telescope_with_obstruction,
    make_two_stage_telescope,
)


def _sensor_xy(camera, rb):
    """(x, y) of intersected sensor positions, for PSF-spread comparisons."""
    sensor_rays, _ = intersect_sensor(camera, rb)
    return sensor_rays.origins[:, 0], sensor_rays.origins[:, 1]


class TestBasicRendering:
    """Telescope.render returns a LazyRayBundle that camera methods consume."""

    def test_render_returns_lazy_bundle(self):
        from iactrace import LazyRayBundle, RayBundle

        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        rb = tel.render(sources, values, source_type="point")
        assert isinstance(rb, LazyRayBundle)

        # Calling materialise() once and inspecting the result.
        flat = rb.materialise()
        assert isinstance(flat, RayBundle)
        assert flat.origins.ndim == 2
        assert flat.directions.ndim == 2
        assert flat.values.ndim == 1
        assert flat.path_length.ndim == 1

    def test_camera_image_shape_and_nonzero(self):
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = cam.image(tel.render(sources, values, source_type="point"))

        assert image.shape == (1, 100, 100)
        assert jnp.sum(image) > 0  # on-axis source produces flux

    def test_parallel_rays_converge_at_center(self):
        """Parallel rays (on-axis) should focus at image center within precision."""
        from iactrace.camera.camera import intersect_sensor

        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, -1.0]])
        rb = tel.render(sources, jnp.array([1.0]), source_type="parallel")
        sensor_rays, _ = intersect_sensor(cam, rb.materialise())

        assert jnp.std(sensor_rays.origins[:, 0]) < 1e-8
        assert jnp.std(sensor_rays.origins[:, 1]) < 1e-8

    def test_camera_collect(self):
        """camera.collect materialises a lazy bundle and returns 4-tuple."""
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        rb = tel.render(sources, values, source_type="point")
        pe_vals, pe_times, pix_id, hit_mask = cam.collect(rb)

        n_rays = pe_vals.shape[0]
        assert pe_times.shape == (n_rays,)
        assert pix_id.shape == (n_rays,)
        assert hit_mask.shape == (n_rays,)
        assert hit_mask.dtype == jnp.bool_


class TestResponseMatrix:
    """Camera.response_matrix(LazyRayBundle)."""

    def test_shape_and_per_source_rows_equal_individual_renders(self):
        tel, cam = make_simple_telescope(n_samples=64)
        # Slightly-off-axis sources avoid pile-ups on the central pixel
        # boundary that would otherwise leak across pixels under pure
        # float-rounding differences between batched and single-source
        # renders. Total throughput matches exactly either way.
        sources = jnp.array(
            [
                [0.0003, 0.0001, -1.0],
                [0.0005, 0.0, -1.0],
                [-0.0005, 0.0, -1.0],
            ]
        )
        single = cam.image(tel.render(sources[:1], jnp.ones(1), source_type="parallel"))
        rm = cam.response_matrix(tel.render(sources, jnp.ones(3), source_type="parallel"))

        # Leading source axis, then each row equals the standalone single-source render.
        assert rm.shape == (sources.shape[0],) + single.shape
        for i in range(sources.shape[0]):
            img_i = cam.image(tel.render(sources[i : i + 1], jnp.ones(1), source_type="parallel"))
            assert jnp.allclose(rm[i], img_i, atol=1e-5), (
                f"Response matrix row {i} disagrees with single-source render."
            )

    def test_rejects_eager_bundle(self):
        """response_matrix needs the per-source structure carried by LazyRayBundle."""
        import pytest

        tel, cam = make_simple_telescope(n_samples=64)
        rb_eager = tel.trace(
            jnp.zeros((10, 3)).at[:, 2].set(50),
            jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (10, 3)),
            jnp.ones(10),
        )
        with pytest.raises(TypeError, match="LazyRayBundle"):
            cam.response_matrix(rb_eager)


class TestRenderLinearity:
    """Linearity / equivalence properties of the fused render path."""

    def test_response_matrix_rows_sum_equals_image(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array(
            [
                [0.0003, 0.0001, -1.0],
                [0.0005, 0.0, -1.0],
                [-0.0005, 0.0, -1.0],
            ]
        )
        rb = tel.render(sources, jnp.ones(3), source_type="parallel")

        assert jnp.allclose(
            cam.response_matrix(rb).sum(axis=0),
            cam.image(rb),
            atol=1e-5,
        )

    def test_lazy_image_matches_materialised_image(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array(
            [
                [0.00036, 0.00036, -1.0],
                [0.00108, 0.00036, -1.0],
                [-0.00036, -0.00108, -1.0],
            ]
        )
        rb = tel.render(sources, jnp.ones(3), source_type="parallel")

        # Lazy fused fold == eager scatter on the materialised flat bundle.
        print(jnp.sum(jnp.abs(cam.image(rb) - cam.image(rb.materialise()))))
        print(cam.image(rb).shape, cam.image(rb.materialise()).shape)
        assert jnp.allclose(cam.image(rb), cam.image(rb.materialise()), atol=1e-5)


class TestEnergyConservation:
    """Test radiometric properties of ray tracing."""

    def test_flux_tracks_mirror_area_and_reflectivity(self):
        """Total collected flux is ~ the mirror collecting area, and scaling
        mirror reflectivity scales the output proportionally."""
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        flux_full = jnp.sum(cam.image(tel.render(sources, values, source_type="point")))
        mirror_area = jnp.pi * 0.1**2
        assert 0.5 * mirror_area < flux_full < 1.5 * mirror_area

        tel_scaled = tel.scale_reflectivity(0, 3)
        flux_scaled = jnp.sum(cam.image(tel_scaled.render(sources, values, source_type="point")))
        assert jnp.isclose(flux_scaled / flux_full, 3.0, rtol=0.01)


class TestMultiStageRendering:
    """Test multi-stage optical systems with multiple mirrors."""

    def test_two_stage_has_correct_stages(self):
        """Two-stage telescope has mirrors in different optical stages."""
        tel, cam = make_two_stage_telescope()

        stages = [g.optical_stage for g in tel.mirror_groups]
        assert stages == [0, 1]
        assert len(tel.mirror_groups) == 2


class TestObstructionEffects:
    """Test that obstructions block light correctly."""

    def test_obstruction_reduces_flux(self):
        """Central obstruction reduces total collected flux."""
        key = jax.random.key(42)

        tel_clear, cam_clear = make_simple_telescope(n_samples=2048, key=key)
        tel_obstructed, cam_obstructed = make_telescope_with_obstruction(n_samples=2048, key=key)

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        image_clear = cam_clear.image(tel_clear.render(sources, values, source_type="parallel"))
        image_obstructed = cam_obstructed.image(
            tel_obstructed.render(sources, values, source_type="parallel")
        )

        flux_clear = jnp.sum(image_clear)
        flux_obstructed = jnp.sum(image_obstructed)

        assert flux_obstructed < flux_clear
        assert flux_obstructed > 0


class TestFinalLegShadow:
    """Shadowing of the converging beam on the last-optic -> focal-plane leg."""

    def _two_rays(self):
        from iactrace.core.ray_bundle import RayBundle

        # Two rays leaving a 'last optic' at z=0 and travelling +z toward a
        # focal plane at z=1: one on the axis, one offset to x=1.
        return RayBundle(
            origins=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            directions=jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            values=jnp.array([1.0, 1.0]),
            path_length=jnp.zeros(2),
            n=jnp.ones(2),
        )

    def test_blocks_ray_crossing_obstruction(self):
        """A ray whose final leg crosses an obstruction is zeroed; a clear
        ray is left untouched, as are its geometry and path length."""
        from iactrace.core.render import apply_final_leg_shadow

        rb = self._two_rays()
        cam_pos = jnp.array([0.0, 0.0, 1.0])  # focal plane at z = 1
        cam_rot = jnp.zeros(3)
        # Sphere squarely on the axial ray's leg, clear of the offset ray.
        sphere = SphereGroup(centers=[[0.0, 0.0, 0.5]], radii=[0.1])

        out = apply_final_leg_shadow(rb, [sphere], cam_pos, cam_rot)

        assert float(out.values[0]) == 0.0  # axial ray blocked
        assert float(out.values[1]) == 1.0  # offset ray untouched
        assert jnp.allclose(out.path_length, rb.path_length)
        assert jnp.allclose(out.origins, rb.origins)
        assert jnp.allclose(out.directions, rb.directions)

    def test_caps_at_focal_plane(self):
        """An obstruction past the focal plane must not shadow; the same
        obstruction is caught once the focal plane is moved beyond it."""
        from iactrace.core.render import apply_final_leg_shadow

        rb = self._two_rays()
        cam_rot = jnp.zeros(3)
        sphere = SphereGroup(centers=[[0.0, 0.0, 1.5]], radii=[0.1])

        # Focal plane at z=1, in front of the sphere -> excluded by the cap.
        out_near = apply_final_leg_shadow(rb, [sphere], jnp.array([0.0, 0.0, 1.0]), cam_rot)
        assert float(out_near.values[0]) == 1.0

        # Focal plane at z=2, past the sphere -> now on the leg, so it blocks.
        out_far = apply_final_leg_shadow(rb, [sphere], jnp.array([0.0, 0.0, 2.0]), cam_rot)
        assert float(out_far.values[0]) == 0.0

    def test_noop_without_obstructions(self):
        """With no obstructions the bundle is returned unchanged."""
        from iactrace.core.render import apply_final_leg_shadow

        rb = self._two_rays()
        out = apply_final_leg_shadow(rb, [], jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3))
        assert out is rb

    def _near_focus_sphere(self):
        # make_simple_telescope: parabola of curvature 1 -> focus at z=0.5,
        # aperture radius 0.1. A small sphere just short of the focus sits
        # deep inside the converging cone but barely clips the incoming beam:
        # its incoming silhouette can remove at most ~(0.02/0.1)**2 = 4% of
        # the flux, so a far larger drop can only come from the final leg.
        return SphereGroup(centers=[[0.0, 0.0, 0.46]], radii=[0.02])

    def test_blocks_converging_cone_in_image(self):
        """Fold path (Camera.image): near-focus obstruction collapses flux.

        The near-focus collapse is asserted once, here on the fold path. The
        materialise (collect) and eager (trace) paths share the same
        apply_final_leg_shadow step, which the unit tests above exercise
        directly, so re-checking the collapse per entry point is redundant."""
        key = jax.random.key(0)
        tel, cam = make_simple_telescope(n_samples=4096, key=key)
        tel_obs = tel.add_obstruction(self._near_focus_sphere())

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        flux_clear = jnp.sum(cam.image(tel.render(sources, values, source_type="parallel")))
        flux_obs = jnp.sum(cam.image(tel_obs.render(sources, values, source_type="parallel")))

        assert flux_clear > 0
        assert flux_obs < 0.2 * flux_clear


class TestRoughnessInTracing:
    """Mirror roughness effects on a real (HESS) telescope loaded from YAML."""

    @pytest.fixture
    def telescope_and_camera(self):
        """Load a test telescope and camera from the split config files."""
        telescope = Telescope.from_yaml("configs/HESS/CT3.yaml", 16, key=jax.random.key(42))
        camera = Camera.from_yaml("configs/HESS/HESS1U.yaml")
        return telescope, camera

    @pytest.fixture
    def test_rays(self):
        """Create test rays for tracing."""
        n_rays = 500
        key1, key2 = jax.random.split(jax.random.key(123))
        r = 5.0 * jnp.sqrt(jax.random.uniform(key1, (n_rays,)))
        theta = jax.random.uniform(key2, (n_rays,)) * 2 * jnp.pi
        origins = jnp.stack(
            [r * jnp.cos(theta), r * jnp.sin(theta), jnp.ones(n_rays) * 100.0], axis=1
        )
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        return origins, directions, jnp.ones(n_rays)

    def test_roughness_increases_psf_spread(self, telescope_and_camera, test_rays):
        """Applying roughness increases the PSF spread."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        rb_clean = telescope.trace(origins, directions, values)
        x_clean, _ = _sensor_xy(camera, rb_clean)

        tel_rough = telescope.apply_roughness(0, 60.0)  # 60 arcsec
        rb_rough = tel_rough.trace(origins, directions, values)
        x_rough, _ = _sensor_xy(camera, rb_rough)

        hit_clean = rb_clean.values > 0
        hit_rough = rb_rough.values > 0
        if jnp.sum(hit_clean) > 10 and jnp.sum(hit_rough) > 10:
            std_clean = jnp.std(x_clean[hit_clean])
            std_rough = jnp.std(x_rough[hit_rough])
            assert std_rough > std_clean, (
                f"Roughness should increase PSF spread: {std_rough} <= {std_clean}"
            )

    def test_zero_roughness_no_change(self, telescope_and_camera, test_rays):
        """Zero roughness does not change the traced positions or values."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        rb_clean = telescope.trace(origins, directions, values)
        x_clean, y_clean = _sensor_xy(camera, rb_clean)

        tel_zero = telescope.apply_roughness(0, 0.0)
        rb_zero = tel_zero.trace(origins, directions, values)
        x_zero, y_zero = _sensor_xy(camera, rb_zero)

        assert jnp.allclose(jnp.stack([x_clean, y_clean], 1), jnp.stack([x_zero, y_zero], 1))
        assert jnp.allclose(rb_clean.values, rb_zero.values)

    def test_roughness_is_deterministic(self, telescope_and_camera, test_rays):
        """Roughness perturbation is deterministic across repeated traces."""
        telescope, camera = telescope_and_camera
        origins, directions, values = test_rays

        tel_rough = telescope.apply_roughness(0, 30.0)
        rb1 = tel_rough.trace(origins, directions, values)
        rb2 = tel_rough.trace(origins, directions, values)
        x1, y1 = _sensor_xy(camera, rb1)
        x2, y2 = _sensor_xy(camera, rb2)

        assert jnp.allclose(jnp.stack([x1, y1], 1), jnp.stack([x2, y2], 1))
        assert jnp.allclose(rb1.values, rb2.values)
