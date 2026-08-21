import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import (
    Camera,
    SellmeierIndex,
    SquareSensorGroup,
    TabulatedIndex,
    TabulatedQE,
    TabulatedResponse,
    Telescope,
)
from iactrace.camera.detector import PMT, ConstantQE
from iactrace.core.apertures import DiskAperture
from iactrace.core.ray_bundle import RayBundle
from iactrace.io import save_camera, save_telescope
from iactrace.telescope.lenses import refractive_group
from iactrace.telescope.mirrors import mirror_group

BK7_B = [[1.03961212, 0.231792344, 1.01046945]]
BK7_C = [[6.00069867e3, 2.00179144e4, 1.03560653e8]]  # nm^2 (um^2 * 1e6)
WL = jnp.array([350.0, 500.0, 650.0])
IDX = jnp.zeros(3, dtype=jnp.int32)


def _detector_bundle(wl):
    n = wl.shape[0]
    return RayBundle(
        origins=jnp.zeros((n, 3)),
        directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1)),
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
        wavelength=wl,
    )


def _save_load_telescope(tel, tmp_path, key):
    fp = tmp_path / "tel.yaml"
    save_telescope(tel, fp)
    return Telescope.from_yaml(fp, 10, key=key)


def _save_load_camera(cam, tmp_path):
    fp = tmp_path / "cam.yaml"
    save_camera(cam, fp)
    return Camera.from_yaml(fp)


def _square(**chain):
    return SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=4,
        height=4,
        bounds=(-0.02, 0.02, -0.02, 0.02),
        **chain,
    )


class TestMirrorCoatingIO:
    def test_wavelength_coating_round_trips(self, tmp_path):
        key = jax.random.key(0)
        coat = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 60.0],
            values=[[0.9, 0.8], [0.7, 0.6]],
            n_elements=1,
            wavelengths=[300.0, 600.0],
        )
        mg = mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.5]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(radii=jnp.array([0.1]), inner_radii=jnp.array([0.0])),
            reflectivity=jnp.array([1.0]),
            reflectivity_curve=coat,
            sample_key=key,
            optical_stage=0,
            n_samples=10,
        )
        tel = Telescope(mirror_groups=[mg], camera_position=[0.0, 0.0, 1.0])
        tel2 = _save_load_telescope(tel, tmp_path, key)
        c2 = tel2.mirror_groups[0].interaction_module.reflectivity_curve
        for cos in (1.0, 0.5):
            np.testing.assert_allclose(
                np.asarray(coat(jnp.full(3, cos), IDX, WL)),
                np.asarray(c2(jnp.full(3, cos), IDX, WL)),
                atol=1e-5,
            )

    def test_angle_only_coating_stays_1d(self, tmp_path):
        # A plain angle curve must serialise as a 1-D `values` list (no
        # `wavelengths_nm`) so existing configs stay byte-identical.
        from iactrace.io.adapters import _curve_to_schema

        coat = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0], values=[0.9, 0.6], n_elements=1
        )
        schema = _curve_to_schema(coat)
        assert schema.wavelengths_nm is None
        assert all(isinstance(x, float) for x in schema.values)


class TestDispersionIO:
    def test_sellmeier_lens_round_trips(self, tmp_path):
        key = jax.random.key(0)
        bk7 = SellmeierIndex(b=jnp.asarray(BK7_B), c=jnp.asarray(BK7_C))
        lg = refractive_group(
            positions=jnp.array([[0.0, 0.0, -0.5]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(radii=jnp.array([0.05]), inner_radii=jnp.array([0.0])),
            index=bk7,
            sample_key=key,
            optical_stage=1,
            n_samples=10,
        )
        tel = Telescope(mirror_groups=[], lens_groups=[lg], camera_position=[0.0, 0.0, 1.0])
        tel2 = _save_load_telescope(tel, tmp_path, key)
        idx2 = tel2.lens_groups[0].interaction_module.index
        assert isinstance(idx2, SellmeierIndex)
        np.testing.assert_allclose(
            np.asarray(bk7.n_at(IDX, WL)), np.asarray(idx2.n_at(IDX, WL)), atol=1e-6
        )

    def test_tabulated_index_lens_round_trips(self, tmp_path):
        key = jax.random.key(1)
        ti = TabulatedIndex.from_table([350.0, 650.0], [1.52, 1.50], n_elements=1)
        lg = refractive_group(
            positions=jnp.array([[0.0, 0.0, -0.5]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(radii=jnp.array([0.05]), inner_radii=jnp.array([0.0])),
            index=ti,
            sample_key=key,
            optical_stage=1,
            n_samples=10,
        )
        tel = Telescope(mirror_groups=[], lens_groups=[lg], camera_position=[0.0, 0.0, 1.0])
        tel2 = _save_load_telescope(tel, tmp_path, key)
        idx2 = tel2.lens_groups[0].interaction_module.index
        assert isinstance(idx2, TabulatedIndex)
        np.testing.assert_allclose(
            np.asarray(ti.n_at(IDX, WL)), np.asarray(idx2.n_at(IDX, WL)), atol=1e-6
        )


class TestQEIO:
    def test_tabulated_qe_round_trips(self, tmp_path):
        qe = TabulatedQE.from_table([300.0, 400.0, 500.0], [0.1, 0.3, 0.25])
        cam = Camera([_square(photodetector=qe, gap=0.0)])
        cam2 = _save_load_camera(cam, tmp_path)
        det2 = cam2.sensor_groups[0].chain.photodetector
        assert isinstance(det2, TabulatedQE)
        rb = _detector_bundle(WL)
        np.testing.assert_allclose(
            np.asarray(qe.detect(rb).values), np.asarray(det2.detect(rb).values), atol=1e-6
        )

    def test_constant_qe_stays_scalar(self, tmp_path):
        cam = Camera([_square(photodetector=ConstantQE(0.85), gap=0.0)])
        det2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.photodetector
        assert isinstance(det2, ConstantQE)
        assert float(det2.qe) == 0.85


class TestConcentratorCoatingIO:
    def test_wall_coating_round_trips(self, tmp_path):
        from iactrace import WinstonCone

        curve = TabulatedResponse.from_wavelengths([300.0, 500.0], [0.5, 1.0], n_elements=1)
        cone = WinstonCone(6, 0.025, 0.01, reflectivity=0.95, reflectivity_curve=curve)
        cam = Camera([_square(concentrator=cone, gap=0.003)])
        cone2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.concentrator
        assert cone2.reflectivity_curve is not None
        cos = jnp.full(WL.shape, 0.6)
        np.testing.assert_allclose(
            np.asarray(cone.wall_reflectivity(cos, WL)),
            np.asarray(cone2.wall_reflectivity(cos, WL)),
            atol=1e-6,
        )

    def test_angle_dependent_wall_coating_round_trips(self, tmp_path):
        from iactrace import WinstonCone

        # The walls see every incidence angle, so the angle axis of the coating
        # has to survive the YAML round trip, not just the wavelength axis.
        curve = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=[[0.4, 0.6], [0.6, 0.8], [0.9, 1.0]],
            n_elements=1,
            wavelengths=[300.0, 700.0],
        )
        cone = WinstonCone(6, 0.025, 0.01, reflectivity=0.95, reflectivity_curve=curve)
        cam = Camera([_square(concentrator=cone, gap=0.003)])
        cone2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.concentrator
        for cos in (1.0, 0.7, 0.2):
            cosines = jnp.full(WL.shape, cos)
            np.testing.assert_allclose(
                np.asarray(cone.wall_reflectivity(cosines, WL)),
                np.asarray(cone2.wall_reflectivity(cosines, WL)),
                atol=1e-6,
            )
        # ... and the reloaded cone is still genuinely angle-dependent.
        normal = np.asarray(cone2.wall_reflectivity(jnp.ones(WL.shape), WL))
        grazing = np.asarray(cone2.wall_reflectivity(jnp.zeros(WL.shape), WL))
        assert np.all(grazing > normal + 1e-3)

    def test_plain_cone_has_no_coating(self, tmp_path):
        from iactrace import WinstonCone

        cone = WinstonCone(6, 0.025, 0.01, reflectivity=0.9)
        cam = Camera([_square(concentrator=cone, gap=0.003)])
        cone2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.concentrator
        assert cone2.reflectivity_curve is None


class TestPMTIO:
    def test_qe_curve_and_dispersive_window_round_trip(self, tmp_path):
        qe = TabulatedResponse.from_wavelengths([300.0, 500.0], [0.1, 0.4], n_elements=1)
        bk7 = SellmeierIndex(b=jnp.asarray(BK7_B), c=jnp.asarray(BK7_C))
        pmt = PMT(qe=0.9, qe_curve=qe, window_index=bk7, face_radius=0.02)
        cam = Camera([_square(photodetector=pmt, gap=0.0)])
        pmt2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.photodetector
        assert isinstance(pmt2, PMT)
        assert pmt2.qe_curve is not None and isinstance(pmt2.window_index, SellmeierIndex)
        rb = _detector_bundle(WL)
        np.testing.assert_allclose(
            np.asarray(pmt.detect(rb).values), np.asarray(pmt2.detect(rb).values), atol=1e-6
        )

    def test_scalar_window_round_trips_as_a_number(self, tmp_path):
        # A non-dispersive window is written as the plain `index` number and
        # comes back as the constant case of the same one field.
        from iactrace.core.refractive_index import ConstantIndex

        pmt = PMT(qe=0.25, window_index=1.48, face_radius=0.02)
        cam = Camera([_square(photodetector=pmt, gap=0.0)])
        assert cam.to_dict()["sensors"][0]["photodetector"]["window_index"] == pytest.approx(1.48)
        pmt2 = _save_load_camera(cam, tmp_path).sensor_groups[0].chain.photodetector
        assert isinstance(pmt2.window_index, ConstantIndex)
        assert float(pmt2.window_index.values[0]) == pytest.approx(1.48)
