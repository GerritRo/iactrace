import equinox as eqx
import jax.numpy as jnp
import pytest

from iactrace import (
    Camera,
    Concentrator,
    DetectionChain,
    HexagonalSensorGroup,
    PhotoSensor,
    SquareSensorGroup,
    UniformQE,
)
from iactrace.camera._hexgeom import _hex_norm, _rotate
from iactrace.camera.camera import intersect_sensor
from iactrace.core.ray_bundle import RayBundle

from .test_integration import make_simple_telescope
from .test_sensors import make_hex_centers

# Stub chain elements (concrete physics is out of scope for the scaffolding)


class StubCone(Concentrator):
    """Minimal concentrator: shift to z=length, attenuate, add path length."""

    length: float = eqx.field(static=True)
    refl: float = eqx.field(static=True)

    def __init__(self, length: float = 0.05, refl: float = 0.9) -> None:
        self.length = float(length)
        self.refl = float(refl)

    def apply(self, local_rays: RayBundle) -> RayBundle:
        o = local_rays.origins
        new_o = jnp.stack([o[:, 0], o[:, 1], o[:, 2] - self.length], axis=-1)
        return RayBundle(
            origins=new_o,
            directions=local_rays.directions,
            values=local_rays.values * self.refl,
            path_length=local_rays.path_length + self.length,
            n=local_rays.n,
        )

    def cross_sections(self):
        angles = jnp.deg2rad(30.0 + 60.0 * jnp.arange(6))
        unit = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (6, 2)
        z = jnp.array([0.0, -self.length])
        rings = jnp.stack([0.02 * unit, 0.01 * unit], axis=0)  # (2, 6, 2)
        return z, rings


class StubPMT(PhotoSensor):
    """Minimal photosensor: attenuate by pde at the detector plane (no advance)."""

    pde: float = eqx.field(static=True)

    def __init__(self, pde: float = 0.8) -> None:
        self.pde = float(pde)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        return RayBundle(
            origins=local_rays.origins,
            directions=local_rays.directions,
            values=local_rays.values * self.pde,
            path_length=local_rays.path_length,
            n=local_rays.n,
        )

    def outline(self):
        s = 0.008
        return jnp.array([[-s, -s], [s, -s], [s, s], [-s, s]])


def _square_sensor(concentrator=None, photosensor=None, gap=0.0):
    return SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]], rotations=[[0.0, 0.0, 0.0]],
        width=8, height=8, bounds=(-1.0, 1.0, -1.0, 1.0),
        concentrator=concentrator, photosensor=photosensor, gap=gap,
    )


def _downward_rays(xy):
    xy = jnp.asarray(xy, dtype=float)
    n = xy.shape[0]
    origins = jnp.concatenate([xy, jnp.ones((n, 1))], axis=-1)
    dirs = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
    return RayBundle(origins=origins, directions=dirs,
                     values=jnp.ones(n), path_length=jnp.zeros(n), n=jnp.ones(n))


# 1. to_pixel_frame correctness


class TestToPixelFrame:
    def test_square_center_maps_to_origin(self):
        sensor = _square_sensor()
        # dx = dy = 0.25; pixel centres at x0 + (i+0.5)*dx = -1 + 0.25*(i+0.5)
        cx = -1.0 + 0.25 * 2.5  # pixel index 2 centre = -0.375
        rays = _downward_rays([[cx, cx]])
        local = sensor.to_pixel_frame(rays, jnp.zeros(1, int))
        assert jnp.allclose(local.origins[:, :2], 0.0, atol=1e-7)
        assert jnp.allclose(local.origins[:, 2], 0.0)

    def test_square_isometry_preserves_values_and_path(self):
        sensor = _square_sensor()
        rays = RayBundle(
            origins=jnp.array([[0.13, -0.42, 0.0]]),
            directions=jnp.array([[0.0, 0.0, 1.0]]),
            values=jnp.array([0.7]),
            path_length=jnp.array([3.5]),
            n=jnp.ones(1),
        )
        local = sensor.to_pixel_frame(rays, jnp.zeros(1, int))
        assert jnp.allclose(local.values, rays.values)
        assert jnp.allclose(local.path_length, rays.path_length)
        # offset must be within half a pixel of the centre
        assert jnp.all(jnp.abs(local.origins[0, 0]) <= sensor.dx / 2 + 1e-6)
        assert jnp.all(jnp.abs(local.origins[0, 1]) <= sensor.dy / 2 + 1e-6)

    def test_hex_rotated_grid_direction_and_offset(self):
        # Build a hex grid rotated by ~12 degrees so grid_rotation != 0.
        theta = jnp.deg2rad(12.0)
        centers = make_hex_centers(n_rings=2, hex_size=0.01)
        cx, cy = _rotate(centers[:, 0], centers[:, 1], theta)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]], rotations=[[0.0, 0.0, 0.0]],
            hex_centers=jnp.stack([cx, cy], axis=-1),
        )
        gr = sensor.grid_rotation
        assert gr > 1e-3  # rotation actually detected

        # Ray at the grid origin (a hex centre) with a known in-plane direction.
        origin = jnp.array([[float(sensor.grid_offset[0]),
                             float(sensor.grid_offset[1]), 0.0]])
        rays = RayBundle(
            origins=origin,
            directions=jnp.array([[1.0, 0.0, 0.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        local = sensor.to_pixel_frame(rays, jnp.zeros(1, int))
        # Centre maps to origin.
        assert jnp.allclose(local.origins[:, :2], 0.0, atol=1e-6)
        # Direction rotated by -grid_rotation: (cos gr, -sin gr).
        assert jnp.allclose(
            local.directions[0, :2],
            jnp.array([jnp.cos(gr), -jnp.sin(gr)]), atol=1e-6,
        )

    def test_hex_in_pixel_local_radius_bounded(self):
        centers = make_hex_centers(n_rings=2, hex_size=0.01)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]], rotations=[[0.0, 0.0, 0.0]],
            hex_centers=centers,
        )
        # Ray near a hex centre stays inside the hexagon (hex_norm <= 1).
        offset = jnp.array([[0.3 * sensor.hex_inradius, 0.0, 0.0]])
        rays = RayBundle(
            origins=offset, directions=jnp.array([[0.0, 0.0, 1.0]]),
            values=jnp.array([1.0]), path_length=jnp.array([0.0]), n=jnp.ones(1),
        )
        local = sensor.to_pixel_frame(rays, jnp.zeros(1, int))
        norm = _hex_norm(local.origins[0, 0], local.origins[0, 1], sensor.hex_inradius)
        assert float(norm) <= 1.0


# 2. Backward compatibility: no concentrator + UniformQE


class TestBackwardCompat:
    def test_image_matches_entrance_binning(self):
        qe = 0.42
        sensor = _square_sensor(photosensor=UniformQE(qe))
        cam = Camera([sensor])
        rays = _downward_rays([[0.1, 0.1], [0.3, -0.2], [-0.5, 0.4], [5.0, 5.0]])

        sensor_rays, s_idx, _ = intersect_sensor(cam, rays)
        expected = sensor.accumulate(
            s_idx, sensor_rays.origins[:, 0], sensor_rays.origins[:, 1],
            sensor_rays.values * qe,
        )
        assert jnp.allclose(cam.image(rays), expected)

    def test_collect_times_and_values_unchanged(self):
        qe = 0.6
        sensor = _square_sensor(photosensor=UniformQE(qe))
        cam = Camera([sensor])
        rays = _downward_rays([[0.1, 0.1], [0.3, -0.2], [5.0, 5.0]])

        sensor_rays, _, _ = intersect_sensor(cam, rays)
        pe, t, _pix, hit = cam.collect(rays)
        # detected rays carry value*qe; the off-sensor (5,5) ray is undetected -> 0.
        assert jnp.allclose(pe, jnp.where(hit, sensor_rays.values * qe, 0.0))
        assert bool(hit[0]) and bool(hit[1]) and not bool(hit[2])
        assert jnp.allclose(t, sensor_rays.path_length)

    def test_collect_pe_sum_matches_image_with_edge_width(self):
        # collect() must honour edge_width exactly like image(): per-ray pe (now
        # zeroed in the dead-zone) sums to the binned image total.
        qe = 0.8
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 0.0]], rotations=[[0.0, 0.0, 0.0]],
            width=8, height=8, bounds=(-1.0, 1.0, -1.0, 1.0),
            edge_width=0.1, photosensor=UniformQE(qe),
        )
        cam = Camera([sensor])
        g = jnp.linspace(-0.95, 0.95, 22)
        xy = jnp.stack(jnp.meshgrid(g, g), axis=-1).reshape(-1, 2)
        rays = _downward_rays(xy)

        pe, _t, _pix, _hit = cam.collect(rays)
        assert jnp.allclose(pe.sum(), cam.image(rays).sum(), atol=1e-5)
        assert 0.0 < float(pe.sum()) < xy.shape[0] * qe  # dead-zone has teeth

    def test_response_matrix_rows_sum_to_image(self):
        # The fused per-source fold (response_matrix) and the plain image both
        # flow through the refactored `_project_to_sensor`; their totals must
        # still agree. Off-axis sources avoid the central-pixel boundary
        # pile-up that float rounding would otherwise leak across pixels.
        tel, cam = make_simple_telescope(n_samples=64)
        sources = jnp.array([[0.0003, 0.0001, -1.0],
                             [0.0005, 0.0, -1.0],
                             [-0.0005, 0.0, -1.0]])
        rb = tel.render(sources, jnp.ones(3), source_type="parallel")
        assert jnp.allclose(cam.response_matrix(rb).sum(axis=0), cam.image(rb), atol=1e-5)


# 3. DetectionChain composition + end-to-end


class TestDetectionChain:
    def test_propagate_composes_cone_and_pmt(self):
        chain = DetectionChain(concentrator=StubCone(length=0.05, refl=0.9),
                               photosensor=StubPMT(pde=0.8), gap=0.01)
        # Rays travel toward -z (canonical light direction): the cone exits them
        # at z=-length and the chain free-flights a further gap to detector_z.
        rays = RayBundle(
            origins=jnp.zeros((3, 3)),
            directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (3, 1)),
            values=jnp.ones(3),
            path_length=jnp.zeros(3),
            n=jnp.ones(3),
        )
        out = chain.propagate(rays)
        assert jnp.allclose(out.values, 0.9 * 0.8)
        assert jnp.allclose(out.path_length, 0.05 + 0.01)

    def test_no_concentrator_is_detect_only(self):
        chain = DetectionChain(concentrator=None, photosensor=StubPMT(pde=0.5), gap=0.02)
        # Rays already at the entrance plane (z=0): with no cone the chain
        # free-flights them across the gap, adding exactly `gap` to path_length.
        rays = RayBundle(
            origins=jnp.array([[0.0, 0.0, 0.0]]),
            directions=jnp.array([[0.0, 0.0, -1.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        out = chain.propagate(rays)
        assert jnp.allclose(out.values, 0.5)
        assert jnp.allclose(out.path_length, 0.02)

    def test_detector_z_single_source_of_truth(self):
        # No concentrator: detector sits at -gap.
        chain = DetectionChain(concentrator=None, photosensor=UniformQE(1.0), gap=0.02)
        assert jnp.allclose(chain.detector_z, -0.02)
        # With a cone: detector sits at -(length + gap).
        chain2 = DetectionChain(concentrator=StubCone(length=0.05),
                                photosensor=UniformQE(1.0), gap=0.02)
        assert jnp.allclose(chain2.detector_z, -(0.05 + 0.02))

    def test_negative_gap_rejected_everywhere(self):
        # __check_init__ on DetectionChain guards every construction path.
        with pytest.raises(ValueError, match="gap"):
            DetectionChain(concentrator=None, photosensor=UniformQE(1.0), gap=-0.5)
        with pytest.raises(ValueError, match="gap"):
            _square_sensor(gap=-0.5)
        with pytest.raises(ValueError, match="gap"):
            Camera([_square_sensor()]).set_gap(0, -0.5)
        # gap == 0.0 stays valid
        assert DetectionChain(concentrator=None, photosensor=UniformQE(1.0), gap=0.0).gap == 0.0

    def test_advance_is_finite_for_parallel_rays(self):
        # A ray parallel to the detector plane (dz=0) never reaches it; the chain
        # leaves it in place so path_length stays finite (it is masked later).
        chain = DetectionChain(concentrator=None, photosensor=UniformQE(1.0), gap=0.02)
        rays = RayBundle(
            origins=jnp.array([[0.0, 0.0, 0.0]]),
            directions=jnp.array([[1.0, 0.0, 0.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        out = chain.propagate(rays)
        assert jnp.all(jnp.isfinite(out.path_length))
        assert jnp.allclose(out.path_length, 0.0)

    def test_camera_end_to_end_with_chain(self):
        sensor = _square_sensor(concentrator=StubCone(refl=0.9),
                                photosensor=StubPMT(pde=0.8), gap=0.01)
        cam = Camera([sensor])
        rays = _downward_rays([[0.1, 0.1], [0.3, -0.2]])
        img = cam.image(rays)
        assert img.shape == (1, 8, 8)
        assert jnp.allclose(img.sum(), 2 * 0.9 * 0.8)
        pe, t, pix, hit = cam.collect(rays)
        assert jnp.allclose(pe, 0.9 * 0.8)
        assert jnp.allclose(t, 1.0 + 0.05 + 0.01)  # drop to z=0, cone (-0.05), gap (+0.01)
        assert bool(hit.all())


# 4. show_sensor_chain viz smoke test


class TestShowSensorChain:
    def test_with_cone_and_detector(self):
        pytest.importorskip("trimesh")
        from iactrace import show_sensor_chain

        cam = Camera([_square_sensor(concentrator=StubCone(),
                                     photosensor=StubPMT(), gap=0.01)])
        scene = show_sensor_chain(cam)
        # entrance + cone walls + detector
        assert len(scene.geometry) >= 3

    def test_without_concentrator(self):
        pytest.importorskip("trimesh")
        from iactrace import show_sensor_chain

        cam = Camera([_square_sensor()])  # no cone, UniformQE
        scene = show_sensor_chain(cam)
        assert len(scene.geometry) >= 1  # entrance (+ detector), no crash

    def test_hexagonal_entrance(self):
        pytest.importorskip("trimesh")
        from iactrace import show_sensor_chain

        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]], rotations=[[0.0, 0.0, 0.0]],
            hex_centers=make_hex_centers(n_rings=2, hex_size=0.01),
            concentrator=StubCone(),
        )
        scene = show_sensor_chain(Camera([sensor]))
        assert len(scene.geometry) >= 2


# 5. Camera setters round-trip through the chain


class TestSetters:
    def test_set_concentrator_and_photosensor(self):
        cam = Camera([_square_sensor()])
        assert cam.sensor_groups[0].chain.concentrator is None
        assert cam.get_info()["sensor_group_0"]["has_concentrator"] is False

        cone = StubCone(length=0.05, refl=0.9)
        cam2 = cam.set_concentrator(0, cone)
        # tree_at rebuilds all-static modules, so compare type + fields, not id.
        chain2 = cam2.sensor_groups[0].chain
        assert isinstance(chain2.concentrator, StubCone)
        assert chain2.concentrator.length == cone.length
        assert chain2.concentrator.refl == cone.refl
        assert cam2.get_info()["sensor_group_0"]["has_concentrator"] is True
        # original is unchanged (functional update)
        assert cam.sensor_groups[0].chain.concentrator is None

        pmt = StubPMT(pde=0.7)
        cam3 = cam2.set_photosensor(0, pmt)
        chain3 = cam3.sensor_groups[0].chain
        assert isinstance(chain3.photosensor, StubPMT)
        assert chain3.photosensor.pde == pmt.pde
        assert isinstance(chain3.concentrator, StubCone)

    def test_clear_concentrator(self):
        cam = Camera([_square_sensor(concentrator=StubCone())])
        cleared = cam.set_concentrator(0, None)
        assert cleared.sensor_groups[0].chain.concentrator is None

    def test_setters_target_one_group_independently(self):
        # The chain lives per group; a Camera setter touches only its sensor_idx.
        cam = Camera([_square_sensor(), _square_sensor()])
        cam2 = cam.set_concentrator(1, StubCone(length=0.05))
        assert cam2.sensor_groups[0].chain.concentrator is None
        assert isinstance(cam2.sensor_groups[1].chain.concentrator, StubCone)
        # functional update leaves the original camera untouched
        assert cam.sensor_groups[1].chain.concentrator is None

    def test_per_group_chains_are_independent(self):
        # Two groups in one camera carry different chains (set at construction).
        g_cone = _square_sensor(concentrator=StubCone(), photosensor=StubPMT(pde=0.7))
        g_plain = _square_sensor(photosensor=UniformQE(0.5))
        cam = Camera([g_cone, g_plain])
        assert cam.sensor_groups[0].chain.concentrator is not None
        assert cam.sensor_groups[1].chain.concentrator is None
        assert cam.get_info()["sensor_group_1"]["has_concentrator"] is False

    def test_set_gap_round_trip(self):
        cam = Camera([_square_sensor(concentrator=StubCone(length=0.05))])
        assert cam.sensor_groups[0].chain.gap == 0.0
        cam2 = cam.set_gap(0, 0.03)
        assert cam2.sensor_groups[0].chain.gap == 0.03
        # functional update leaves the original untouched
        assert cam.sensor_groups[0].chain.gap == 0.0
        # the detector plane reflects cone length + gap
        assert jnp.allclose(cam2.sensor_groups[0].chain.detector_z, -(0.05 + 0.03))

    def test_get_info_reports_gap_and_detector_z(self):
        cam = Camera([_square_sensor(concentrator=StubCone(length=0.05), gap=0.01)])
        info = cam.get_info()["sensor_group_0"]
        assert jnp.allclose(info["gap"], 0.01)
        assert jnp.allclose(info["detector_z"], -(0.05 + 0.01))


# 6. Concentrator fill index -> optical-path-length weighting


class TestConcentratorIndex:
    def test_default_index_is_air(self):
        # Hollow guides (Winston cones, the StubCone) inherit the base air index.
        assert StubCone().index == 1.0

    def test_index_override_weights_opl(self):
        # A solid dielectric guide overrides `index`; its `apply` weights the
        # internal geometric path by that index when accumulating OPL.
        class SolidStub(Concentrator):
            length: float = eqx.field(static=True)
            n_mat: float = eqx.field(static=True)

            def __init__(self, length: float = 0.05, n_mat: float = 1.5) -> None:
                self.length = float(length)
                self.n_mat = float(n_mat)

            @property
            def index(self) -> float:
                return self.n_mat

            def apply(self, local_rays: RayBundle) -> RayBundle:
                o = local_rays.origins
                new_o = jnp.stack([o[:, 0], o[:, 1], o[:, 2] - self.length], axis=-1)
                return RayBundle(
                    origins=new_o,
                    directions=local_rays.directions,
                    values=local_rays.values,
                    path_length=local_rays.path_length + self.index * self.length,
                    n=local_rays.n,
                )

        rays = RayBundle(
            origins=jnp.zeros((1, 3)),
            directions=jnp.array([[0.0, 0.0, -1.0]]),
            values=jnp.ones(1), path_length=jnp.zeros(1), n=jnp.ones(1),
        )
        out = SolidStub(length=0.05, n_mat=1.5).apply(rays)
        # OPL = index * geometric length, not the bare geometric 0.05.
        assert jnp.allclose(out.path_length, 1.5 * 0.05)
