import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from iactrace import PMT, AngularResponse, Camera, RayBundle, TabulatedResponse
from iactrace.camera.detector import landing_coordinates
from iactrace.io import (
    Include,
    IncludeDumper,
    YAMLConfigError,
    build_camera_config,
    load_yaml,
    resolve_includes,
)

from ._helpers import spherical_cap_surface

R_DOME = 0.021
DOWN = jnp.array([0.0, 0.0, -1.0])


def _dome_points(rho, phi=0.0):
    """Points on, and outward normals of, a sphere of radius R_DOME with its apex at z = 0."""
    rho = np.asarray(rho, dtype=float)
    s = rho / R_DOME
    points = np.stack([rho * np.cos(phi), rho * np.sin(phi), R_DOME * (np.sqrt(1 - s**2) - 1)], -1)
    normals = np.stack([s * np.cos(phi), s * np.sin(phi), np.sqrt(1 - s**2)], axis=-1)
    return jnp.asarray(points), jnp.asarray(normals)


def _unit(v):
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return jnp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True))


def _table():
    """Response linear in every axis: 1 + i/100 + 10 r + a/1000 + (wl - 300)/1000."""
    inc, rad = np.array([0.0, 30.0, 60.0]), np.array([0.0, 0.006, 0.015])
    az, wl = np.array([0.0, 90.0, 180.0]), np.array([300.0, 400.0, 600.0])
    i, r, a, w = np.meshgrid(inc, rad, az, wl, indexing="ij")
    values = 1 + i / 100 + 10 * r + a / 1000 + (w - 300) / 1000
    return AngularResponse(inc, values, radius=rad, azimuth_deg=az, wavelengths_nm=wl)


def _rays(points, directions, wavelength=400.0):
    n = points.shape[0]
    return RayBundle(
        origins=points,
        directions=jnp.broadcast_to(directions, (n, 3)),
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
        wavelength=jnp.full(n, wavelength),
    )


class TestLandingCoordinates:
    def test_axis_parallel_rays_on_a_dome(self):
        rho = np.array([0.004, 0.009, 0.014])
        points, normals = _dome_points(rho, phi=0.7)
        inc, radius, az = landing_coordinates(points, jnp.tile(DOWN, (3, 1)), normals)
        np.testing.assert_allclose(radius, rho, rtol=1e-5)
        np.testing.assert_allclose(inc, np.degrees(np.arcsin(rho / R_DOME)), atol=1e-3)
        np.testing.assert_allclose(az, 0.0, atol=0.05)

    def test_azimuth_distinguishes_outward_from_toward_the_axis(self):
        point, normal = _dome_points([0.010])  # on +x, normal tilted 28.4 deg toward +x
        inc_out, _, az_out = landing_coordinates(point, _unit([1.0, 0.0, -1.0]), normal)
        inc_in, _, az_in = landing_coordinates(point, _unit([-1.0, 0.0, -1.0]), normal)
        assert float(az_out[0]) == pytest.approx(0.0, abs=0.05)
        assert float(az_in[0]) == pytest.approx(180.0, abs=0.05)
        assert float(inc_out[0]) == pytest.approx(45.0 + 28.437, abs=0.01)
        assert float(inc_in[0]) == pytest.approx(45.0 - 28.437, abs=0.01)
        _, _, az_side = landing_coordinates(point, _unit([0.0, 1.0, -1.0]), normal)
        assert 0.0 < float(az_side[0]) < 90.0

    def test_flat_face_has_a_position_and_an_azimuth(self):
        points = jnp.array([[0.0, 0.008, 0.0]])
        up = jnp.array([[0.0, 0.0, 1.0]])
        inc, radius, az = landing_coordinates(points, _unit([0.0, 1.0, -1.0]), up)
        assert float(radius[0]) == pytest.approx(0.008)
        assert float(inc[0]) == pytest.approx(45.0, abs=1e-3)
        assert float(az[0]) == pytest.approx(0.0, abs=0.05)  # heading outward
        _, _, az_in = landing_coordinates(points, _unit([0.0, -1.0, -1.0]), up)
        _, _, az_side = landing_coordinates(points, _unit([1.0, 0.0, -1.0]), up)
        assert float(az_in[0]) == pytest.approx(180.0, abs=0.05)
        assert float(az_side[0]) == pytest.approx(90.0, abs=0.05)

    def test_gradient_is_finite_on_the_axis_at_normal_incidence(self):
        table = _table()

        def response(tilt):
            d = jnp.stack([jnp.sin(tilt), 0.0, -jnp.cos(tilt)])[None]
            return table.at(jnp.zeros((1, 3)), d, jnp.array([[0.0, 0.0, 1.0]]), jnp.array([400.0]))[
                0
            ]

        assert np.isfinite(float(jax.grad(response)(0.0)))


class TestAngularResponse:
    def test_multilinear_interpolation_and_clamping(self):
        got = _table()(
            jnp.array([15.0, 80.0]),
            jnp.array([0.010, 0.030]),
            jnp.array([45.0, 180.0]),
            jnp.array([500.0, 900.0]),
        )
        inside = 1 + 15 / 100 + 10 * 0.010 + 45 / 1000 + 200 / 1000
        clamped = 1 + 60 / 100 + 10 * 0.015 + 180 / 1000 + 300 / 1000
        np.testing.assert_allclose(got, [inside, clamped], rtol=1e-5)

    def test_an_axis_that_is_not_given_carries_no_dependence(self):
        table = AngularResponse([0.0, 60.0], [1.0, 2.0])
        assert table.values.shape == (2, 1, 1, 1)
        got = table(jnp.array([30.0]), jnp.array([0.01]), jnp.array([120.0]), jnp.array([650.0]))
        assert float(got[0]) == pytest.approx(1.5)

    def test_shape_and_ordering_are_checked(self):
        with pytest.raises(ValueError, match="shape"):
            AngularResponse([0.0, 60.0], [1.0])
        with pytest.raises(ValueError, match="shape"):
            AngularResponse([0.0, 60.0], [1.0, 2.0], radius=[0.0, 0.01])
        with pytest.raises(ValueError, match="increasing"):
            AngularResponse([60.0, 0.0], [1.0, 2.0])


class TestPMTResponseFunctions:
    """A PMT multiplies qe, qe_curve and angular_response; a missing one is 1."""

    def _detect(self, points, **functions):
        surface = spherical_cap_surface(0.015, 0.0063)  # sphere of radius 21 mm
        pmt = PMT(qe=0.5, face_radius=0.015, surface=surface, vertex_z=0.0, **functions)
        return np.asarray(pmt.detect(_rays(points, DOWN)).values)

    def test_each_function_multiplies_and_a_missing_one_is_one(self):
        rho = np.array([0.002, 0.006, 0.011])
        points, _ = _dome_points(rho, phi=-1.2)
        theta = np.degrees(np.arcsin(rho / R_DOME))  # incidence angle of the axis-parallel rays
        qe_curve = TabulatedResponse.from_wavelengths([300.0, 500.0], [0.2, 0.4], n_elements=1)
        # relative responses, deliberately above 1
        angle_only = AngularResponse([0.0, 90.0], [1.0, 3.0])
        f_qe = 0.3
        f_angle_only = 1.0 + 2.0 * theta / 90.0
        f_full = 1 + theta / 100 + 10 * rho + 0.1

        base = self._detect(points)
        np.testing.assert_allclose(base, 0.5, rtol=1e-6)
        np.testing.assert_allclose(self._detect(points, qe_curve=qe_curve), 0.5 * f_qe, rtol=1e-5)
        np.testing.assert_allclose(
            self._detect(points, angular_response=angle_only), 0.5 * f_angle_only, rtol=5e-4
        )
        np.testing.assert_allclose(
            self._detect(points, angular_response=_table()), 0.5 * f_full, rtol=5e-4
        )
        everything = self._detect(points, qe_curve=qe_curve, angular_response=_table())
        np.testing.assert_allclose(everything, 0.5 * f_qe * f_full, rtol=1e-3)

    def test_angular_response_on_a_flat_pmt(self):
        pmt = PMT(qe=1.0, face_radius=0.015, angular_response=_table())
        points = jnp.array([[0.003, 0.0, 0.0], [0.0, -0.012, 0.0]])
        got = np.asarray(pmt.detect(_rays(points, _unit([0.0, 0.0, -1.0])[0])).values)
        # normal incidence: azimuth undefined (90 deg by convention), radius still resolves
        np.testing.assert_allclose(got, 1 + 10 * np.array([0.003, 0.012]) + 0.09 + 0.1, rtol=2e-4)


def _camera_config(photodetector):
    return {
        "sensors": [
            {
                "id": "s",
                "type": "square",
                "width": 10,
                "height": 10,
                "bounds": [-0.005, 0.005, -0.005, 0.005],
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "photodetector": photodetector,
            }
        ]
    }


def _response_config():
    return {
        "type": "table",
        "angles_deg": [0.0, 60.0],
        "radius": [0.0, 0.015],
        "azimuth_deg": [0.0, 180.0],
        "wavelengths_nm": [300.0, 600.0],
        "values": np.linspace(0.5, 2.0, 16).reshape(2, 2, 2, 2).tolist(),
    }


class TestPMTResponseConfig:
    def test_roundtrip_through_config(self):
        pmt = {
            "type": "pmt",
            "qe": 0.3,
            "face_radius": 0.015,
            "angular_response": _response_config(),
        }
        camera = build_camera_config(_camera_config(pmt))
        detector = camera.sensor_groups[0].chain.photodetector
        assert detector.qe_curve is None
        assert detector.angular_response.values.shape == (2, 2, 2, 2)
        dumped = camera.to_dict()["sensors"][0]["photodetector"]
        assert "qe_curve" not in dumped
        np.testing.assert_allclose(
            dumped["angular_response"]["values"], _response_config()["values"], rtol=1e-6
        )
        np.testing.assert_allclose(dumped["angular_response"]["radius"], [0.0, 0.015], rtol=1e-6)

    def test_angle_only_table_roundtrips_without_the_other_axes(self):
        angular = {"type": "table", "angles_deg": [0.0, 40.0, 80.0], "values": [1.0, 1.2, 1.9]}
        pmt = {"type": "pmt", "face_radius": 0.015, "angular_response": angular}
        camera = build_camera_config(_camera_config(pmt))
        dumped = camera.to_dict()["sensors"][0]["photodetector"]["angular_response"]
        np.testing.assert_allclose(dumped["values"], [1.0, 1.2, 1.9], rtol=1e-6)
        np.testing.assert_allclose(dumped["angles_deg"], [0, 40, 80], atol=1e-3)
        assert not {"radius", "azimuth_deg", "wavelengths_nm"} & set(dumped)

    def test_qe_curve_stays_bounded_by_one(self):
        too_big = {"type": "table", "angles_deg": [0.0, 80.0], "values": [1.0, 1.5]}
        with pytest.raises(YAMLConfigError, match=r"\[0, 1\]"):
            build_camera_config(
                _camera_config({"type": "pmt", "face_radius": 0.015, "qe_curve": too_big})
            )

    @pytest.mark.parametrize(
        "change",
        [
            {"angles_deg": [0.0, 60.0, 80.0]},  # shape mismatch
            {"azimuth_deg": [0.0, 200.0]},  # out of range
            {"radius": [0.015, 0.0]},  # not increasing
            {"values": (-np.ones((2, 2, 2, 2))).tolist()},  # negative
        ],
    )
    def test_invalid_tables_are_rejected(self, change):
        pmt = {
            "type": "pmt",
            "face_radius": 0.015,
            "angular_response": {**_response_config(), **change},
        }
        with pytest.raises(YAMLConfigError):
            build_camera_config(_camera_config(pmt))


class TestInclude:
    def _write(self, tmp_path):
        (tmp_path / "shared").mkdir()
        (tmp_path / "shared" / "response.yaml").write_text(yaml.safe_dump(_response_config()))
        (tmp_path / "shared" / "pmt.yaml").write_text(
            "type: pmt\nqe: 0.3\nface_radius: 0.015\nangular_response: !include response.yaml\n"
        )
        config = _camera_config(Include("shared/pmt.yaml"))
        camera_file = tmp_path / "camera.yaml"
        camera_file.write_text(yaml.dump(config, Dumper=IncludeDumper, sort_keys=False))
        return camera_file

    def test_nested_includes_resolve_relative_to_the_including_file(self, tmp_path):
        camera_file = self._write(tmp_path)
        assert "!include" in camera_file.read_text()
        pmt = load_yaml(camera_file)["sensors"][0]["photodetector"]
        assert pmt["qe"] == 0.3
        assert pmt["angular_response"]["radius"] == [0.0, 0.015]
        detector = Camera.from_yaml(camera_file).sensor_groups[0].chain.photodetector
        assert detector.angular_response.values.shape == (2, 2, 2, 2)

    def test_unresolved_includes_survive_an_edit_and_save(self, tmp_path):
        camera_file = self._write(tmp_path)
        config = load_yaml(camera_file, resolve_includes=False)
        assert config["sensors"][0]["photodetector"] == Include("shared/pmt.yaml")
        config["sensors"][0]["id"] = "renamed"
        camera_file.write_text(yaml.dump(config, Dumper=IncludeDumper, sort_keys=False))
        reloaded = load_yaml(camera_file)
        assert reloaded["sensors"][0]["id"] == "renamed"
        assert reloaded == resolve_includes(config, tmp_path)

    def test_merge_include_with_local_overrides(self, tmp_path):
        (tmp_path / "response.yaml").write_text(yaml.safe_dump(_response_config()))
        (tmp_path / "pmt.yaml").write_text(
            "type: pmt\nqe: 0.3\nface_radius: 0.015\nangular_response: !include response.yaml\n"
        )
        camera_file = tmp_path / "camera.yaml"
        config = _camera_config({"<<": Include("pmt.yaml"), "qe": 0.9, "vertex_z": 0.004})
        camera_file.write_text(yaml.dump(config, Dumper=IncludeDumper, sort_keys=False))
        assert "    <<: !include" in camera_file.read_text()  # a plain merge key

        pmt = load_yaml(camera_file)["sensors"][0]["photodetector"]
        assert pmt["qe"] == 0.9  # the local key wins
        assert pmt["face_radius"] == 0.015 and pmt["vertex_z"] == 0.004
        assert "<<" not in pmt
        detector = Camera.from_yaml(camera_file).sensor_groups[0].chain.photodetector
        assert detector.qe == pytest.approx(0.9)
        assert detector.angular_response.values.shape == (2, 2, 2, 2)

        kept = load_yaml(camera_file, resolve_includes=False)
        assert kept["sensors"][0]["photodetector"]["<<"] == Include("pmt.yaml")
        assert resolve_includes(kept, tmp_path) == load_yaml(camera_file)

    def test_merge_include_needs_a_mapping(self, tmp_path):
        (tmp_path / "list.yaml").write_text("[1, 2]\n")
        (tmp_path / "a.yaml").write_text("x:\n  <<: !include list.yaml\n  y: 1\n")
        with pytest.raises(YAMLConfigError, match="mapping"):
            load_yaml(tmp_path / "a.yaml")

    def test_missing_and_circular_includes_raise(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: !include missing.yaml\n")
        with pytest.raises(YAMLConfigError, match="not found"):
            load_yaml(tmp_path / "a.yaml")
        (tmp_path / "b.yaml").write_text("x: !include c.yaml\n")
        (tmp_path / "c.yaml").write_text("y: !include b.yaml\n")
        with pytest.raises(YAMLConfigError, match="Circular"):
            load_yaml(tmp_path / "b.yaml")
