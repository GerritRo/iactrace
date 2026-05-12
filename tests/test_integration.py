import jax
import jax.numpy as jnp

from iactrace import Camera, SquareSensorGroup, Telescope
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.obstructions import CylinderGroup
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import AsphericSurfaceGroup


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
    interaction = ReflectInteraction(reflectivity=jnp.ones(n))
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


def make_simple_telescope(curvature=1.0, n_samples=1024, key=None):
    """Create a minimal telescope + camera for testing."""
    if key is None:
        key = jax.random.key(0)

    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([curvature])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = _make_disk_mirror_group(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0, n_samples=n_samples,
    )

    focal_length = 1.0 / (2.0 * curvature) if curvature != 0 else 1000.0

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )

    telescope = Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=None,
        name="test_telescope",
        camera_position=[0.0, 0.0, focal_length],
    )

    camera = Camera(sensor_groups=[sensor])

    return telescope, camera


def make_two_stage_telescope(n_samples=512, key=None):
    """Create a two-stage telescope + camera."""
    if key is None:
        key = jax.random.key(0)

    first = _make_disk_mirror_group(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        rotations=jnp.array([[0.0, 0.0, 0.0]]),
        curvatures=jnp.array([0.0]),
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.1]),
        optical_stage=0, n_samples=n_samples,
    )

    second = _make_disk_mirror_group(
        positions=jnp.array([[0.0, 0.0, 0.5]]),
        rotations=jnp.array([[0.0, 45.0, 0.0]]),
        curvatures=jnp.array([0.0]),
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.2]),
        optical_stage=1, n_samples=n_samples,
    )

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=50,
        height=50,
        bounds=(-0.2, 0.2, -0.2, 0.2),
    )

    telescope = Telescope(
        mirror_groups=[first, second],
        obstruction_groups=None,
        name="two_stage_telescope",
        camera_position=[0.5, 0.0, 0.5],
        camera_rotation=[0.0, 90.0, 0.0],
    )

    camera = Camera(sensor_groups=[sensor])

    return telescope, camera


def make_telescope_with_obstruction(n_samples=1024, key=None):
    """Create a telescope + camera with a central obstruction."""
    if key is None:
        key = jax.random.key(0)

    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([1.0])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = _make_disk_mirror_group(
        positions, rotations, curvatures, conics, aspherics, radii,
        optical_stage=0, n_samples=n_samples,
    )

    obstruction = CylinderGroup(
        p1=[[0.0, 0.0, 0.1]],
        p2=[[0.0, 0.0, 0.4]],
        r=[0.03],
    )

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )

    telescope = Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=[obstruction],
        name="obstructed_telescope",
        camera_position=[0.0, 0.0, 0.5],
    )

    camera = Camera(sensor_groups=[sensor])

    return telescope, camera


class TestBasicRendering:
    """Telescope.render returns a LazyRayBundle that camera methods consume."""

    def test_render_returns_lazy_bundle(self):
        from iactrace import LazyRayBundle, RayBundle

        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        rb = tel.render(sources, values, source_type='point')
        assert isinstance(rb, LazyRayBundle)

        # Calling materialise() once and inspecting the result.
        flat = rb.materialise()
        assert isinstance(flat, RayBundle)
        assert flat.origins.ndim == 2
        assert flat.directions.ndim == 2
        assert flat.values.ndim == 1
        assert flat.path_length.ndim == 1

    def test_camera_image_returns_correct_shape(self):
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = cam.image(tel.render(sources, values, source_type='point'))

        assert image.shape == (1, 100, 100)

    def test_render_nonzero_for_on_axis_source(self):
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = cam.image(tel.render(sources, values, source_type='point'))

        assert jnp.sum(image) > 0

    def test_parallel_rays_converge_at_center(self):
        """Parallel rays (on-axis) should focus at image center within precision."""
        from iactrace.camera.camera import intersect_sensor

        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, -1.0]])
        rb = tel.render(sources, jnp.array([1.0]), source_type='parallel')
        sensor_rays, _, _ = intersect_sensor(cam, rb.materialise())

        assert jnp.std(sensor_rays.origins[:, 0]) < 1e-8
        assert jnp.std(sensor_rays.origins[:, 1]) < 1e-8

    def test_camera_collect(self):
        """camera.collect materialises a lazy bundle and returns 4-tuple."""
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        rb = tel.render(sources, values, source_type='point')
        pe_vals, pe_times, pix_id, hit_mask = cam.collect(rb)

        n_rays = pe_vals.shape[0]
        assert pe_times.shape == (n_rays,)
        assert pix_id.shape == (n_rays,)
        assert hit_mask.shape == (n_rays,)
        assert hit_mask.dtype == jnp.bool_


class TestResponseMatrix:
    """Camera.response_matrix(LazyRayBundle) — fused per-source response."""

    def test_shape_matches_image_with_leading_source_axis(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array([
            [0.0, 0.0, -1.0],
            [0.001, 0.0, -1.0],
            [-0.001, 0.0, -1.0],
        ])
        single = cam.image(tel.render(sources[:1], jnp.ones(1), source_type='parallel'))

        rm = cam.response_matrix(tel.render(sources, jnp.ones(3), source_type='parallel'))

        assert rm.shape == (sources.shape[0],) + single.shape

    def test_per_source_rows_equal_individual_renders(self):
        tel, cam = make_simple_telescope(n_samples=64)
        # Slightly-off-axis sources avoid pile-ups on the central pixel
        # boundary that would otherwise leak across pixels under pure
        # float-rounding differences between batched and single-source
        # renders. Total throughput matches exactly either way.
        sources = jnp.array([
            [0.0003, 0.0001, -1.0],
            [0.0005, 0.0, -1.0],
            [-0.0005, 0.0, -1.0],
        ])

        rm = cam.response_matrix(
            tel.render(sources, jnp.ones(3), source_type='parallel'),
        )

        for i in range(sources.shape[0]):
            img_i = cam.image(
                tel.render(sources[i:i + 1], jnp.ones(1), source_type='parallel'),
            )
            assert jnp.allclose(rm[i], img_i, atol=1e-5), (
                f"Response matrix row {i} disagrees with single-source render."
            )

    def test_values_weight_rows(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array([
            [0.0, 0.0, -1.0],
            [0.0005, 0.0, -1.0],
        ])

        unit = cam.response_matrix(
            tel.render(sources, jnp.ones(2), source_type='parallel'),
        )
        weighted = cam.response_matrix(
            tel.render(sources, jnp.array([2.0, 3.0]), source_type='parallel'),
        )

        assert jnp.allclose(weighted[0], 2.0 * unit[0], atol=1e-5)
        assert jnp.allclose(weighted[1], 3.0 * unit[1], atol=1e-5)

    def test_rejects_eager_bundle(self):
        """response_matrix needs the per-source structure carried by LazyRayBundle."""
        import pytest

        tel, cam = make_simple_telescope(n_samples=64)
        rb_eager = tel.trace(
            jnp.zeros((10, 3)).at[:, 2].set(50),
            jnp.broadcast_to(jnp.array([0., 0., -1.]), (10, 3)),
            jnp.ones(10),
        )
        with pytest.raises(TypeError, match="LazyRayBundle"):
            cam.response_matrix(rb_eager)


class TestRenderLinearity:
    """Linearity / equivalence properties of the fused render path."""

    def test_response_matrix_rows_sum_equals_image(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array([
            [0.0003, 0.0001, -1.0],
            [0.0005, 0.0, -1.0],
            [-0.0005, 0.0, -1.0],
        ])
        rb = tel.render(sources, jnp.ones(3), source_type='parallel')

        assert jnp.allclose(
            cam.response_matrix(rb).sum(axis=0),
            cam.image(rb),
            atol=1e-5,
        )

    def test_lazy_image_matches_materialised_image(self):
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array([
            [0.0003, 0.0001, -1.0],
            [0.0005, 0.0, -1.0],
            [-0.0005, 0.0, -1.0],
        ])
        rb = tel.render(sources, jnp.ones(3), source_type='parallel')

        # Lazy fused fold == eager scatter on the materialised flat bundle.
        assert jnp.allclose(cam.image(rb), cam.image(rb.materialise()), atol=1e-5)


class TestEnergyConservation:
    """Test radiometric properties of ray tracing."""

    def test_output_scales_with_input_intensity(self):
        """Output flux scales linearly with input intensity."""
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])

        values1 = jnp.array([1.0])
        values2 = jnp.array([3.0])

        image1 = cam.image(tel.render(sources, values1, source_type='point'))
        image2 = cam.image(tel.render(sources, values2, source_type='point'))

        ratio = jnp.sum(image2) / jnp.sum(image1)
        assert jnp.isclose(ratio, 3.0, rtol=0.01)

    def test_output_scales_with_mirror_area(self):
        """Output flux is proportional to mirror collecting area."""
        tel, cam = make_simple_telescope()
        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image = cam.image(tel.render(sources, values, source_type='point'))
        total_flux = jnp.sum(image)

        mirror_area = jnp.pi * 0.1**2

        assert total_flux > 0.5 * mirror_area * values[0]
        assert total_flux < 1.5 * mirror_area * values[0]

    def test_reflectivity_scales_output(self):
        """Scaling mirror reflectivity scales output proportionally."""
        tel, cam = make_simple_telescope()

        sources = jnp.array([[0.0, 0.0, 1e6]])
        values = jnp.array([1.0])

        image_full = cam.image(tel.render(sources, values, source_type='point'))
        flux_full = jnp.sum(image_full)

        tel_scaled = tel.scale_reflectivity(0, 3)
        image_scaled = cam.image(tel_scaled.render(sources, values, source_type='point'))
        flux_scaled = jnp.sum(image_scaled)

        assert jnp.isclose(flux_scaled / flux_full, 3.0, rtol=0.01)


class TestMultiStageRendering:
    """Test multi-stage optical systems with multiple mirrors."""

    def test_two_stage_has_correct_stages(self):
        """Two-stage telescope has mirrors in different optical stages."""
        tel, cam = make_two_stage_telescope()

        stages = [g.optical_stage for g in tel.mirror_groups]
        assert stages == [0, 1]
        assert len(tel.mirror_groups) == 2

    def test_single_stage_produces_output(self):
        """Single-stage telescope works correctly (baseline)."""
        tel, cam = make_simple_telescope()

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        image = cam.image(tel.render(sources, values, source_type='parallel'))
        assert jnp.sum(image) > 0


class TestObstructionEffects:
    """Test that obstructions block light correctly."""

    def test_obstruction_reduces_flux(self):
        """Central obstruction reduces total collected flux."""
        key = jax.random.key(42)

        tel_clear, cam_clear = make_simple_telescope(n_samples=2048, key=key)
        tel_obstructed, cam_obstructed = make_telescope_with_obstruction(n_samples=2048, key=key)

        sources = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])

        image_clear = cam_clear.image(
            tel_clear.render(sources, values, source_type='parallel')
        )
        image_obstructed = cam_obstructed.image(
            tel_obstructed.render(sources, values, source_type='parallel')
        )

        flux_clear = jnp.sum(image_clear)
        flux_obstructed = jnp.sum(image_obstructed)

        assert flux_obstructed < flux_clear
        assert flux_obstructed > 0

    def test_obstruction_group_is_attached(self):
        """Telescope with obstruction has the obstruction group attached."""
        tel, cam = make_telescope_with_obstruction()

        assert tel.obstruction_groups is not None
        assert len(tel.obstruction_groups) == 1
        assert len(tel.obstruction_groups[0]) == 1
