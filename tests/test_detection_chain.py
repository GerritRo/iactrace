import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import (
    Camera,
    Concentrator,
    ConstantQE,
    DetectionChain,
    HexagonalSensorGroup,
    PhotoDetector,
    SquareSensorGroup,
)
from iactrace.camera._hexgeom import _hex_norm, _rotate
from iactrace.camera.camera import intersect_sensor
from iactrace.core.ray_bundle import RayBundle

from ._helpers import bin_positions, make_hex_centers, spherical_cap_surface

# Stub chain elements (concrete physics is out of scope for the scaffolding)


class StubCone(Concentrator):
    """Minimal concentrator: shift to z=-length, attenuate, add path, then land."""

    length: float = eqx.field(static=True)
    refl: float = eqx.field(static=True)

    def __init__(self, length: float = 0.05, refl: float = 0.9) -> None:
        self.length = float(length)
        self.refl = float(refl)

    def to_surface(self, local_rays: RayBundle, surface) -> RayBundle:
        o = local_rays.origins
        shifted = local_rays.replace(
            origins=jnp.stack([o[:, 0], o[:, 1], o[:, 2] - self.length], axis=-1),
            values=local_rays.values * self.refl,
            path_length=local_rays.path_length + self.length,
        )
        return surface.stop(shifted)

    def cross_sections(self):
        angles = jnp.deg2rad(30.0 + 60.0 * jnp.arange(6))
        unit = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (6, 2)
        z = jnp.array([0.0, -self.length])
        rings = jnp.stack([0.02 * unit, 0.01 * unit], axis=0)  # (2, 6, 2)
        return z, rings


class StubPMT(PhotoDetector):
    """Minimal photodetector: attenuate by pde at the detector plane (no advance)."""

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


def _square_sensor(concentrator=None, photodetector=None, gap=0.0):
    return SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=8,
        height=8,
        bounds=(-1.0, 1.0, -1.0, 1.0),
        concentrator=concentrator,
        photodetector=photodetector,
        gap=gap,
    )


def _downward_rays(xy):
    xy = jnp.asarray(xy, dtype=float)
    n = xy.shape[0]
    origins = jnp.concatenate([xy, jnp.ones((n, 1))], axis=-1)
    dirs = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
    return RayBundle(
        origins=origins,
        directions=dirs,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


# 1. to_pixel_frame correctness


def _localize(sensor, rays):
    """Assign pixels (single tile) and reframe -- the pipeline's two-step."""
    n = rays.origins.shape[0]
    pix_id, valid = sensor.pixel_index_and_mask(
        jnp.zeros(n, int), rays.origins[:, 0], rays.origins[:, 1]
    )
    return sensor.to_pixel_frame(rays, pix_id), pix_id, valid


class TestToPixelFrame:
    def test_square_isometry_preserves_values_and_path(self):
        sensor = _square_sensor()
        rays = RayBundle(
            origins=jnp.array([[0.13, -0.42, 0.0]]),
            directions=jnp.array([[0.0, 0.0, 1.0]]),
            values=jnp.array([0.7]),
            path_length=jnp.array([3.5]),
            n=jnp.ones(1),
        )
        local, _, _ = _localize(sensor, rays)
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
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=jnp.stack([cx, cy], axis=-1),
        )
        gr = sensor.grid_rotation
        assert gr > 1e-3  # rotation actually detected

        # Ray at the grid origin (a hex centre) with a known in-plane direction.
        origin = jnp.array([[float(sensor.grid_offset[0]), float(sensor.grid_offset[1]), 0.0]])
        rays = RayBundle(
            origins=origin,
            directions=jnp.array([[1.0, 0.0, 0.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        local, _, _ = _localize(sensor, rays)
        # Centre maps to origin.
        assert jnp.allclose(local.origins[:, :2], 0.0, atol=1e-6)
        # Direction rotated by -grid_rotation: (cos gr, -sin gr).
        assert jnp.allclose(
            local.directions[0, :2],
            jnp.array([jnp.cos(gr), -jnp.sin(gr)]),
            atol=1e-6,
        )

    def test_hex_frame_matches_binning(self):
        # The frame a ray is reframed into and the pixel it is binned into come
        # from ONE pixel_index_and_mask call; on a rotated layout every valid
        # ray must land inside its assigned hexagon, and the centre table must
        # reconstruct the grid coordinates exactly.
        theta = jnp.deg2rad(17.0)
        centers = make_hex_centers(n_rings=3, hex_size=0.01)
        gx, gy = _rotate(centers[:, 0], centers[:, 1], theta)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=jnp.stack([gx, gy], axis=-1),
        )
        rng = np.random.default_rng(7)
        xy = rng.uniform(-0.08, 0.08, (500, 2))
        rays = _downward_rays(xy)
        local, pix_id, valid = _localize(sensor, rays)

        v = np.asarray(valid)
        assert 100 < v.sum() < v.size  # both populations present
        norms = _hex_norm(local.origins[:, 0], local.origins[:, 1], sensor.hex_inradius)
        assert np.all(np.asarray(norms)[v] <= 1.0 + 1e-9)

        # local + assigned centre reproduces the grid coordinates (pure isometry)
        x_grid, y_grid = sensor._to_grid_coords(rays.origins[:, 0], rays.origins[:, 1])
        c = np.asarray(sensor.pixel_centers_grid)[np.asarray(pix_id) % sensor.n_pixels]
        assert np.allclose(np.asarray(local.origins[:, 0]) + c[:, 0], np.asarray(x_grid))
        assert np.allclose(np.asarray(local.origins[:, 1]) + c[:, 1], np.asarray(y_grid))


# 2. Backward compatibility: no concentrator + ConstantQE


class TestBackwardCompat:
    def test_image_matches_entrance_binning(self):
        # A no-cone ConstantQE chain must reproduce the old entrance-binning image
        # exactly (a released-file compatibility guarantee).
        qe = 0.42
        sensor = _square_sensor(photodetector=ConstantQE(qe))
        cam = Camera([sensor])
        rays = _downward_rays([[0.1, 0.1], [0.3, -0.2], [-0.5, 0.4], [5.0, 5.0]])

        sensor_rays, s_idx = intersect_sensor(cam, rays)
        expected = bin_positions(
            sensor,
            s_idx,
            sensor_rays.origins[:, 0],
            sensor_rays.origins[:, 1],
            sensor_rays.values * qe,
        )
        assert jnp.allclose(cam.image(rays), expected)

    def test_collect_pe_sum_matches_image_with_edge_width(self):
        # Kept separate: this exercises the edge_width dead-zone in collect(),
        # a different code path from the plain entrance-binning above.
        # collect() must honour edge_width exactly like image(): per-ray pe (now
        # zeroed in the dead-zone) sums to the binned image total.
        qe = 0.8
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=8,
            height=8,
            bounds=(-1.0, 1.0, -1.0, 1.0),
            edge_width=0.1,
            photodetector=ConstantQE(qe),
        )
        cam = Camera([sensor])
        g = jnp.linspace(-0.95, 0.95, 22)
        xy = jnp.stack(jnp.meshgrid(g, g), axis=-1).reshape(-1, 2)
        rays = _downward_rays(xy)

        pe, _t, _pix, _hit = cam.collect(rays)
        assert jnp.allclose(pe.sum(), cam.image(rays).sum(), atol=1e-5)
        assert 0.0 < float(pe.sum()) < xy.shape[0] * qe  # dead-zone has teeth


# 3. DetectionChain composition + end-to-end


class TestDetectionChain:
    def test_propagate_composes_cone_and_pmt(self):
        chain = DetectionChain(
            concentrator=StubCone(length=0.05, refl=0.9), photodetector=StubPMT(pde=0.8), gap=0.01
        )
        # Rays travel toward -z (canonical light direction): the cone exits them
        # at z=-length and the chain free-flights a further gap to detector_z.
        rays = RayBundle(
            origins=jnp.zeros((3, 3)),
            directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (3, 1)),
            values=jnp.ones(3),
            path_length=jnp.zeros(3),
            n=jnp.ones(3),
        )
        out = chain.propagate(rays).rays
        assert jnp.allclose(out.values, 0.9 * 0.8)
        assert jnp.allclose(out.path_length, 0.05 + 0.01)

    def test_no_concentrator_is_detect_only(self):
        chain = DetectionChain(concentrator=None, photodetector=StubPMT(pde=0.5), gap=0.02)
        # Rays already at the entrance plane (z=0): with no cone the chain
        # free-flights them across the gap, adding exactly `gap` to path_length.
        rays = RayBundle(
            origins=jnp.array([[0.0, 0.0, 0.0]]),
            directions=jnp.array([[0.0, 0.0, -1.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        out = chain.propagate(rays).rays
        assert jnp.allclose(out.values, 0.5)
        assert jnp.allclose(out.path_length, 0.02)

    def test_negative_gap_rejected_everywhere(self):
        # __check_init__ on DetectionChain guards every construction path.
        with pytest.raises(ValueError, match="gap"):
            DetectionChain(concentrator=None, photodetector=ConstantQE(1.0), gap=-0.5)
        with pytest.raises(ValueError, match="gap"):
            _square_sensor(gap=-0.5)
        with pytest.raises(ValueError, match="gap"):
            Camera([_square_sensor()]).set_gap(0, -0.5)
        # gap == 0.0 stays valid
        assert DetectionChain(concentrator=None, photodetector=ConstantQE(1.0), gap=0.0).gap == 0.0

    def test_advance_is_finite_for_parallel_rays(self):
        # A ray parallel to the detector plane (dz=0) never reaches it; the chain
        # leaves it in place so path_length stays finite (it is masked later).
        chain = DetectionChain(concentrator=None, photodetector=ConstantQE(1.0), gap=0.02)
        rays = RayBundle(
            origins=jnp.array([[0.0, 0.0, 0.0]]),
            directions=jnp.array([[1.0, 0.0, 0.0]]),
            values=jnp.array([1.0]),
            path_length=jnp.array([0.0]),
            n=jnp.ones(1),
        )
        out = chain.propagate(rays).rays
        assert jnp.all(jnp.isfinite(out.path_length))
        assert jnp.allclose(out.path_length, 0.0)

    def test_camera_end_to_end_with_chain(self):
        sensor = _square_sensor(
            concentrator=StubCone(refl=0.9), photodetector=StubPMT(pde=0.8), gap=0.01
        )
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

        cam = Camera([_square_sensor(concentrator=StubCone(), photodetector=StubPMT(), gap=0.01)])
        scene = show_sensor_chain(cam)
        # entrance + cone walls + detector
        assert len(scene.geometry) >= 3


# 5. Camera setters round-trip through the chain


class TestSetters:
    def test_set_concentrator_and_photodetector(self):
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
        cam3 = cam2.set_photodetector(0, pmt)
        chain3 = cam3.sensor_groups[0].chain
        assert isinstance(chain3.photodetector, StubPMT)
        assert chain3.photodetector.pde == pmt.pde
        assert isinstance(chain3.concentrator, StubCone)

        # Clearing the concentrator (set to None) restores the detect-only chain.
        cleared = cam3.set_concentrator(0, None)
        assert cleared.sensor_groups[0].chain.concentrator is None

    def test_per_group_chains_are_independent(self):
        # Two groups in one camera carry different chains (set at construction).
        g_cone = _square_sensor(concentrator=StubCone(), photodetector=StubPMT(pde=0.7))
        g_plain = _square_sensor(photodetector=ConstantQE(0.5))
        cam = Camera([g_cone, g_plain])
        assert cam.sensor_groups[0].chain.concentrator is not None
        assert cam.sensor_groups[1].chain.concentrator is None
        assert cam.get_info()["sensor_group_1"]["has_concentrator"] is False


# 6. DetectionSurface: every photodetector has one; generic core shapes match conics


def _slanted_rays(n=200, z0=0.05, slope=0.08, seed=2):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.006, 0.006, (n, 2))
    origins = jnp.asarray(np.c_[xy, np.full(n, z0)])
    d = jnp.tile(jnp.array([slope, 0.0, -1.0]), (n, 1))
    d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    return RayBundle(
        origins=origins,
        directions=d,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


def _sphere_as_core_group(curvature):
    from iactrace.core.surfaces import AsphericSurfaceGroup

    return AsphericSurfaceGroup(
        offsets=jnp.zeros((1, 2)),
        curvatures=jnp.asarray([curvature]),
        conics=jnp.zeros(1),
        aspherics=jnp.zeros((1, 0)),
    )


class TestDetectionSurface:
    def test_every_photodetector_has_a_surface(self):
        from iactrace import PMT
        from iactrace.camera.detector import DetectionSurface

        flat = ConstantQE(0.5).surface
        assert isinstance(flat, DetectionSurface)
        assert flat.is_flat and flat.vertex_z == 0.0

        pmt = PMT(
            qe=1.0, face_radius=0.01, surface=spherical_cap_surface(0.01, 0.002), vertex_z=0.002
        )
        dome = pmt.surface
        assert not dome.is_flat
        assert dome.vertex_z == pytest.approx(0.002)
        assert dome.radius == pytest.approx(0.01)

    def test_shape_and_conic_kwargs_are_exclusive(self):
        from iactrace.camera.detector import DetectionSurface

        with pytest.raises(ValueError, match="not both"):
            DetectionSurface(_sphere_as_core_group(-30.0), curvature=-30.0)

    def test_generic_shape_stop_matches_conic_fast_path(self):
        # The same sphere given as curvature (closed-form intersection) and as a
        # core AsphericSurfaceGroup (Newton-Raphson) must stop rays identically.
        from iactrace.camera.detector import DetectionSurface

        c = -1.0 / 0.03
        fast = DetectionSurface(vertex_z=-0.01, curvature=c, radius=0.02)
        generic = DetectionSurface(_sphere_as_core_group(c), vertex_z=-0.01, radius=0.02)
        rays = _slanted_rays()

        out_f = fast.stop(rays)
        out_g = generic.stop(rays)
        assert jnp.allclose(out_f.origins, out_g.origins, atol=1e-9)
        assert jnp.allclose(out_f.values, out_g.values)
        assert jnp.allclose(out_f.path_length, out_g.path_length, atol=1e-9)
        # normals are photodetector-side now; both descriptions must agree there too
        assert jnp.allclose(
            fast.normals_at(out_f.origins), generic.normals_at(out_g.origins), atol=1e-9
        )

    def test_chain_with_pmt_photocathode(self):
        # End-to-end: cone + curved PMT photocathode + Fresnel window response.
        from iactrace import PMT, WinstonCone

        cone = WinstonCone(6, 0.02, 0.01, reflectivity=0.95, max_bounces=12)
        pmt = PMT(
            qe=0.8,
            window_index=1.48,
            face_radius=0.011,
            surface=spherical_cap_surface(0.011, 0.003),
            vertex_z=0.003,
        )
        chain = DetectionChain(concentrator=cone, photodetector=pmt, gap=0.002)
        out = chain.propagate(_slanted_rays(z0=0.0, slope=0.1)).rays
        v = jnp.asarray(out.values)
        assert float(v.sum()) > 0.0
        # qe and the window transmittance both bite: nothing exceeds qe.
        assert float(v.max()) < 0.8
