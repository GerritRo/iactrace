import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Camera, ConstantQE, SquareSensorGroup, Telescope, WinstonCone
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from iactrace.io import (
    build_camera_config,
    build_telescope_config,
    save_camera,
    save_telescope,
    telescope_to_dict,
)
from iactrace.io.yaml_io import YAMLConfigError
from iactrace.telescope import operations as ops


@pytest.fixture
def simple_disk_telescope_config():
    return {
        "telescope": {
            "name": "simple_disk",
            "units": "m",
            "camera_position": [0.0, 0.0, 10.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": {
            "spherical": {
                "surface": {
                    "type": "aspheric",
                    "curvature": 0.05,
                    "conic": -1.0,
                    "aspheric": [],
                }
            }
        },
        "mirrors": [
            {
                "id": "M_0",
                "template": "spherical",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "aperture": {"type": "circular", "radius": 0.5},
            },
            {
                "id": "M_1",
                "template": "spherical",
                "position": [1.0, 0.0, 0.0],
                "orientation": [0.0, 5.0, 0.0],
                "aperture": {"type": "circular", "radius": 0.5},
            },
        ],
        "obstructions": [],
    }


@pytest.fixture
def polygon_telescope_config():
    return {
        "telescope": {
            "name": "polygon_telescope",
            "units": "m",
            "camera_position": [0.0, 0.0, 15.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": {
            "hex_surface": {
                "surface": {
                    "type": "aspheric",
                    "curvature": 0.033,
                    "conic": 0.0,
                    "aspheric": [],
                }
            }
        },
        "mirrors": [
            {
                "id": "M_0",
                "template": "hex_surface",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "aperture": {
                    "type": "polygon",
                    "vertices": [
                        [0.0, 0.52],
                        [0.45, 0.26],
                        [0.45, -0.26],
                        [0.0, -0.52],
                        [-0.45, -0.26],
                        [-0.45, 0.26],
                    ],
                },
            },
        ],
        "obstructions": [],
    }


@pytest.fixture
def telescope_with_obstructions_config():
    return {
        "telescope": {
            "name": "with_obstructions",
            "units": "m",
            "camera_position": [0.0, 0.0, 10.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": {
            "primary": {"surface": {"type": "aspheric", "curvature": 0.05, "conic": -1.0, "aspheric": []}}
        },
        "mirrors": [
            {
                "id": "M_0",
                "template": "primary",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "aperture": {"type": "circular", "radius": 0.5},
            },
        ],
        "obstructions": [
            {
                "id": "cylinder_0",
                "type": "cylinder",
                "p1": [0.0, 0.0, 0.0],
                "p2": [0.0, 0.0, 5.0],
                "r": 0.1,
            },
            {"id": "box_0", "type": "box", "p1": [-0.5, -0.5, 2.0], "p2": [0.5, 0.5, 2.5]},
            {"id": "sphere_0", "type": "sphere", "center": [1.0, 0.0, 3.0], "r": 0.2},
        ],
    }


# Conversion + saving


class TestTelescopeToDict:
    """telescope_to_dict produces a telescope-only dict."""

    def test_conversion_preserves_structure(
        self, n_samples, random_key, simple_disk_telescope_config
    ):
        telescope = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        result = telescope_to_dict(telescope)

        assert "telescope" in result
        assert "mirrors" in result
        assert result["telescope"]["name"] == "simple_disk"
        assert len(result["mirrors"]) == 2
        # Telescope dict must NOT contain detector geometry.
        assert "sensors" not in result
        assert "camera" not in result
        # A plain mirror (no coating/reflectivity) writes neither field.
        for tpl in result["mirror_templates"].values():
            assert "reflectivity" not in tpl and "coating" not in tpl
        for m in result["mirrors"]:
            assert "reflectivity" not in m


class TestSaveTelescope:
    def test_save_overwrite_false_raises(
        self, n_samples, random_key, simple_disk_telescope_config, tmp_path
    ):
        telescope = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope, filepath)
        with pytest.raises(FileExistsError):
            save_telescope(telescope, filepath, overwrite=False)
        save_telescope(telescope, filepath, overwrite=True)  # overwrite=True succeeds


# Round-trip


class TestRoundTrip:
    def test_disk_mirror_roundtrip(
        self, n_samples, random_key, simple_disk_telescope_config, tmp_path
    ):
        telescope1 = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        assert telescope1.name == telescope2.name
        assert len(telescope1.mirror_groups) == len(telescope2.mirror_groups)
        for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
            np.testing.assert_allclose(
                np.asarray(g1.positions), np.asarray(g2.positions), rtol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(g1.rotations), np.asarray(g2.rotations), rtol=1e-5
            )

    def _roundtrip_mirror(self, bsdf, reflectivity, random_key, tmp_path):
        import jax.numpy as jnp

        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.mirrors import mirror_group

        mg = mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.5]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(radii=jnp.array([0.1]), inner_radii=jnp.array([0.0])),
            reflectivity=jnp.array([reflectivity]),
            bsdf=bsdf,
            sample_key=random_key,
            optical_stage=0,
            n_samples=10,
        )
        tel = Telescope(mirror_groups=[mg], camera_position=[0.0, 0.0, 1.0])
        filepath = tmp_path / "t.yaml"
        save_telescope(tel, filepath)
        return Telescope.from_yaml(filepath, 10, key=random_key).mirror_groups[0]

    def test_reflectivity_and_bsdf_roundtrip(self, random_key, tmp_path):
        """Mirror reflectivity and both BSDF roughness models survive save/load
        (reflectivity was once silently reset to 1.0)."""
        import jax.numpy as jnp

        from iactrace.core.bsdf import DoubleGaussianBSDF, GaussianBSDF

        # plain reflectivity, no BSDF
        g = self._roundtrip_mirror(bsdf=None, reflectivity=0.83, random_key=random_key, tmp_path=tmp_path)
        np.testing.assert_allclose(np.asarray(g.interaction_module.reflectivity_scalar), [0.83], rtol=1e-5)

        # single Gaussian BSDF
        g = self._roundtrip_mirror(
            bsdf=GaussianBSDF(scale=jnp.array([25.0])), reflectivity=1.0, random_key=random_key, tmp_path=tmp_path
        )
        assert isinstance(g.bsdf, GaussianBSDF)
        np.testing.assert_allclose(np.asarray(g.bsdf.scale), [25.0], rtol=1e-5)

        # double Gaussian BSDF, carried alongside a non-unit reflectivity
        g = self._roundtrip_mirror(
            bsdf=DoubleGaussianBSDF(
                scale_narrow=jnp.array([10.0]), scale_wide=jnp.array([120.0]), mix_weight=jnp.array([0.2])
            ),
            reflectivity=0.9,
            random_key=random_key,
            tmp_path=tmp_path,
        )
        assert isinstance(g.bsdf, DoubleGaussianBSDF)
        np.testing.assert_allclose(np.asarray(g.bsdf.scale_narrow), [10.0], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(g.bsdf.scale_wide), [120.0], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(g.bsdf.mix_weight), [0.2], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(g.interaction_module.reflectivity_scalar), [0.9], rtol=1e-5)

    def test_polygon_mirror_roundtrip(
        self, n_samples, random_key, polygon_telescope_config, tmp_path
    ):
        telescope1 = build_telescope_config(polygon_telescope_config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        from iactrace.core.apertures import PolygonAperture

        for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
            if isinstance(getattr(g1, "aperture", None), PolygonAperture):
                assert isinstance(getattr(g2, "aperture", None), PolygonAperture)
                np.testing.assert_allclose(
                    np.asarray(g1.aperture.vertices),
                    np.asarray(g2.aperture.vertices),
                    rtol=1e-5,
                )

    def test_obstructions_roundtrip(
        self, n_samples, random_key, telescope_with_obstructions_config, tmp_path
    ):
        telescope1 = build_telescope_config(
            telescope_with_obstructions_config, n_samples, random_key
        )
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        assert telescope1.obstruction_groups is not None
        assert telescope2.obstruction_groups is not None
        assert len(telescope1.obstruction_groups) == len(telescope2.obstruction_groups)

    def test_surface_parameters_preserved(self, n_samples, random_key, tmp_path):
        config = {
            "telescope": {
                "name": "aspheric_test",
                "units": "m",
                "camera_position": [0.0, 0.0, 10.0],
                "camera_rotation": [0.0, 0.0, 0.0],
            },
            "mirror_templates": {
                "aspheric": {
                    "surface": {
                        "type": "aspheric",
                        "curvature": 0.0123456,
                        "conic": -1.5,
                        "aspheric": [1e-6, 2e-8, 3e-10],
                    }
                }
            },
            "mirrors": [
                {
                    "id": "M_0",
                    "template": "aspheric",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
            "obstructions": [],
        }
        telescope1 = build_telescope_config(config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath, precision=12)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
            np.testing.assert_allclose(
                np.asarray(g1.surface.curvatures),
                np.asarray(g2.surface.curvatures),
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                np.asarray(g1.surface.conics),
                np.asarray(g2.surface.conics),
                rtol=1e-10,
            )

    def test_hexagonal_sensor_roundtrip(self, tmp_path):
        cam_config = {
            "sensors": [
                {
                    "id": "sensor_0",
                    "type": "hexagonal",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "centers_x": [0.0, 1.0, 2.0, 0.5, 1.5],
                    "centers_y": [0.0, 0.0, 0.0, 0.866, 0.866],
                }
            ],
        }
        camera1 = build_camera_config(cam_config)
        filepath = tmp_path / "t.yaml"
        save_camera(camera1, filepath)
        camera2 = Camera.from_yaml(filepath)

        from iactrace.camera.sensor_group import HexagonalSensorGroup

        s1 = camera1.sensor_groups[0]
        s2 = camera2.sensor_groups[0]
        assert isinstance(s1, HexagonalSensorGroup)
        assert isinstance(s2, HexagonalSensorGroup)
        np.testing.assert_allclose(
            np.asarray(s1.hex_centers),
            np.asarray(s2.hex_centers),
            rtol=1e-5,
        )

    def _square(self, concentrator=None, photodetector=None, gap=0.0):
        return SquareSensorGroup(
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=4,
            height=4,
            bounds=(-0.02, 0.02, -0.02, 0.02),
            concentrator=concentrator,
            photodetector=photodetector,
            gap=gap,
        )

    def test_camera_chain_scalars_roundtrip(self, tmp_path):
        camera1 = Camera([self._square(photodetector=ConstantQE(0.85), gap=0.004)])
        filepath = tmp_path / "t.yaml"
        save_camera(camera1, filepath)
        chain = Camera.from_yaml(filepath).sensor_groups[0].chain
        assert isinstance(chain.photodetector, ConstantQE)
        assert chain.photodetector.qe == pytest.approx(0.85)
        assert chain.gap == pytest.approx(0.004)
        assert chain.concentrator is None

    def test_camera_winston_cone_roundtrip(self, tmp_path):
        cone = WinstonCone(
            n_sides=6,
            entrance_apothem=0.025,
            exit_apothem=0.01,
            reflectivity=0.92,
            max_bounces=8,
            orientation_deg=15.0,
        )
        camera1 = Camera([self._square(concentrator=cone, gap=0.003)])
        filepath = tmp_path / "t.yaml"
        save_camera(camera1, filepath)
        chain2 = Camera.from_yaml(filepath).sensor_groups[0].chain
        c1, c2 = camera1.sensor_groups[0].chain.concentrator, chain2.concentrator
        assert isinstance(c2, WinstonCone)
        assert c2.n_sides == c1.n_sides
        assert c2.max_bounces == c1.max_bounces
        assert c2.exit_apothem == pytest.approx(c1.exit_apothem)
        assert c2.entrance_apothem == pytest.approx(c1.entrance_apothem)
        assert c2.length == pytest.approx(c1.length, rel=1e-4)
        assert c2.reflectivity == pytest.approx(c1.reflectivity)
        assert c2.orientation == pytest.approx(c1.orientation)
        assert chain2.gap == pytest.approx(0.003)

        # A genuinely truncated cone (physical mouth at z = L) also round-trips,
        # exercising the explicit-length serialization branch.
        import math

        import jax.numpy as jnp

        from iactrace.camera.optics.winston import profile_apothem

        a2, cutoff_deg, length = 0.01, 20.0, 0.04
        s, c = math.sin(math.radians(cutoff_deg)), math.cos(math.radians(cutoff_deg))
        phys = float(profile_apothem(jnp.asarray(length), a2, s, c))
        trunc1 = WinstonCone(6, entrance_apothem=phys, exit_apothem=a2, length=length)
        fp2 = tmp_path / "trunc.yaml"
        save_camera(Camera([self._square(concentrator=trunc1)]), fp2, precision=12)
        trunc2 = Camera.from_yaml(fp2).sensor_groups[0].chain.concentrator
        assert isinstance(trunc2, WinstonCone)
        assert trunc2.entrance_apothem == pytest.approx(trunc1.entrance_apothem)
        assert trunc2.length == pytest.approx(trunc1.length)
        assert (trunc2.s, trunc2.c) == pytest.approx((trunc1.s, trunc1.c))

    def test_camera_okumura_cone_roundtrip(self, tmp_path):
        from iactrace import OkumuraCone

        cone = OkumuraCone.cubic(
            n_sides=6,
            entrance_apothem=0.025,
            exit_apothem=0.01,
            p1=(0.39, 0.18),
            p2=(0.87, 0.36),
            reflectivity=0.92,
            max_bounces=8,
            orientation_deg=15.0,
        )
        camera1 = Camera([self._square(concentrator=cone, gap=0.003)])
        filepath = tmp_path / "t.yaml"
        save_camera(camera1, filepath)
        sensor = camera1.to_dict()["sensors"][0]
        assert sensor["concentrator"]["type"] == "okumura"
        chain2 = Camera.from_yaml(filepath).sensor_groups[0].chain
        c1, c2 = camera1.sensor_groups[0].chain.concentrator, chain2.concentrator
        assert isinstance(c2, OkumuraCone)
        assert c2.degree == c1.degree == 3
        assert c2.control_points == pytest.approx(np.asarray(c1.control_points))
        assert c2.n_sides == c1.n_sides
        assert c2.max_bounces == c1.max_bounces
        assert c2.exit_apothem == pytest.approx(c1.exit_apothem)
        assert c2.entrance_apothem == pytest.approx(c1.entrance_apothem)
        assert c2.length == pytest.approx(c1.length, rel=1e-4)
        assert c2.reflectivity == pytest.approx(c1.reflectivity)
        assert c2.orientation == pytest.approx(c1.orientation)
        assert chain2.gap == pytest.approx(0.003)

    def test_camera_pmt_roundtrip(self, tmp_path):
        from iactrace import PMT

        from ._helpers import spherical_cap_surface

        pmt = PMT(
            qe=0.35,
            n_window=1.48,
            face_radius=0.011,
            surface=spherical_cap_surface(0.011, 0.003),
            vertex_z=0.003,
            length=0.05,
            n_facets=32,
        )
        camera1 = Camera([self._square(photodetector=pmt)])
        filepath = tmp_path / "t.yaml"
        save_camera(camera1, filepath)
        sensor = camera1.to_dict()["sensors"][0]
        assert sensor["photodetector"]["type"] == "pmt"
        p2 = Camera.from_yaml(filepath).sensor_groups[0].chain.photodetector
        assert isinstance(p2, PMT)
        assert p2.qe == pytest.approx(pmt.qe)
        assert p2.n_window == pytest.approx(pmt.n_window)
        assert p2.face_radius == pytest.approx(pmt.face_radius)
        assert p2.vertex_z == pytest.approx(pmt.vertex_z)
        assert p2.shape.curvatures[0] == pytest.approx(pmt.shape.curvatures[0])
        assert p2.length == pytest.approx(pmt.length)
        assert p2.n_facets == pmt.n_facets

        # A minimal PMT with length=None resolves to 2*face_radius at
        # construction; the resolved value is written and reloads exactly.
        default_pmt = PMT(qe=0.9, face_radius=0.008)
        fp2 = tmp_path / "pmt2.yaml"
        save_camera(Camera([self._square(photodetector=default_pmt)]), fp2)
        p3 = Camera.from_yaml(fp2).sensor_groups[0].chain.photodetector
        assert isinstance(p3, PMT)
        assert p3.n_window is None
        assert p3.length == pytest.approx(2 * 0.008)

    def test_nonuniform_chain_elements_raise(self, tmp_path):
        """Unrepresentable photodetectors and concentrators must raise instead of
        silently downgrading to ConstantQE(1.0) / no-concentrator."""
        import equinox as eqx

        from iactrace.camera.detector import DetectionSurface, PhotoDetector
        from iactrace.camera.optics import Concentrator
        from iactrace.core.ray_bundle import RayBundle

        class WeirdSensor(PhotoDetector):
            pde: float = eqx.field(static=True)

            def __init__(self, pde=0.5):
                self.pde = float(pde)

            def detect(self, local_rays: RayBundle) -> RayBundle:
                return local_rays.replace(values=local_rays.values * self.pde)

        class WeirdCone(Concentrator):
            length: float = eqx.field(static=True, default=0.01)

            def to_surface(self, rays: RayBundle, surface: DetectionSurface) -> RayBundle:
                return surface.stop(rays)

        fp = tmp_path / "t.yaml"
        with pytest.raises(ValueError, match="WeirdSensor is not representable"):
            save_camera(Camera([self._square(photodetector=WeirdSensor(0.5))]), fp)
        assert not fp.exists()
        with pytest.raises(ValueError, match="WeirdCone is not representable"):
            save_camera(Camera([self._square(concentrator=WeirdCone())]), fp)
        assert not fp.exists()

    def test_sensor_without_chain_keys_backward_compatible(self):
        # Released camera files are `sensors:`-only with no detection-chain keys;
        # each group must load with a trivial chain (no cone, perfect QE, no gap).
        camera = build_camera_config(
            {
                "sensors": [
                    {
                        "type": "square",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                        "width": 4,
                        "height": 4,
                        "bounds": [-0.02, 0.02, -0.02, 0.02],
                    }
                ],
            }
        )
        chain = camera.sensor_groups[0].chain
        assert chain.concentrator is None
        assert isinstance(chain.photodetector, ConstantQE)
        assert chain.photodetector.qe == pytest.approx(1.0)
        assert chain.gap == pytest.approx(0.0)

    def test_photodetector_slot_explicit_yaml(self):
        # The detector response is a first-class discriminated `photodetector:` slot
        # on each sensor group, alongside its `gap`.
        camera = build_camera_config(
            {
                "sensors": [
                    {
                        "type": "square",
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                        "width": 4,
                        "height": 4,
                        "bounds": [-0.02, 0.02, -0.02, 0.02],
                        "photodetector": {"type": "constant", "qe": 0.7},
                        "gap": 0.002,
                    }
                ],
            }
        )
        chain = camera.sensor_groups[0].chain
        assert isinstance(chain.photodetector, ConstantQE)
        assert chain.photodetector.qe == pytest.approx(0.7)
        assert chain.gap == pytest.approx(0.002)

    def test_multiple_stages_preserved(self, n_samples, random_key, tmp_path):
        config = {
            "telescope": {
                "name": "two_stage",
                "units": "m",
                "camera_position": [0.0, 0.0, -1.0],
                "camera_rotation": [0.0, 0.0, 0.0],
            },
            "mirror_templates": {
                "primary": {"surface": {"type": "aspheric", "curvature": 0.05, "conic": -1.0, "aspheric": []}},
                "secondary": {"surface": {"type": "aspheric", "curvature": 0.1, "conic": 0.0, "aspheric": []}},
            },
            "mirrors": [
                {
                    "id": "M_primary",
                    "template": "primary",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "aperture": {"type": "circular", "radius": 1.0},
                    "stage": 0,
                },
                {
                    "id": "M_secondary",
                    "template": "secondary",
                    "position": [0.0, 0.0, 5.0],
                    "orientation": [180.0, 0.0, 0.0],
                    "aperture": {"type": "circular", "radius": 0.2},
                    "stage": 1,
                },
            ],
            "obstructions": [],
        }
        telescope1 = build_telescope_config(config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        assert len(telescope2.mirror_groups) == 2
        stages = {g.optical_stage for g in telescope2.mirror_groups}
        assert 0 in stages
        assert 1 in stages

    def test_reflectivity_curve_roundtrip(self, n_samples, random_key, tmp_path):
        """A tabulated R(theta) curve survives save -> load."""
        from iactrace.core.coatings import (
            TabulatedCoating,
        )

        config = {
            "telescope": {
                "name": "with_curve",
                "units": "m",
                "camera_position": [0.0, 0.0, 10.0],
                "camera_rotation": [0.0, 0.0, 0.0],
            },
            "mirror_templates": {
                "silver": {
                    "surface": {"type": "aspheric", "curvature": 0.05, "conic": -1.0, "aspheric": []},
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 30.0, 60.0, 80.0],
                        "values": [0.96, 0.95, 0.90, 0.60],
                    },
                },
            },
            "mirrors": [
                {
                    "template": "silver",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "aperture": {"type": "circular", "radius": 0.5},
                    "reflectivity": 0.98,
                },
            ],
            "obstructions": [],
        }
        telescope1 = build_telescope_config(config, n_samples, random_key)
        filepath = tmp_path / "t.yaml"
        save_telescope(telescope1, filepath)
        telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

        interaction1 = telescope1.mirror_groups[0].interaction_module
        interaction2 = telescope2.mirror_groups[0].interaction_module

        # Both have a tabulated curve, byte-identical scalar
        assert isinstance(interaction1.reflectivity, TabulatedCoating)
        assert isinstance(interaction2.reflectivity, TabulatedCoating)
        np.testing.assert_allclose(
            np.asarray(interaction1.reflectivity.cos_table),
            np.asarray(interaction2.reflectivity.cos_table),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(interaction1.reflectivity.values),
            np.asarray(interaction2.reflectivity.values),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(interaction1.reflectivity_scalar),
            np.asarray(interaction2.reflectivity_scalar),
            rtol=1e-10,
        )



# =============================================================================
# Zernike figure-surface (de)serialization and round-trips
# =============================================================================

KEY = jax.random.key(0)


def _config(templates, mirrors):
    return {
        "telescope": {
            "name": "z",
            "units": "m",
            "camera_position": [0.0, 0.0, 10.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": templates,
        "mirrors": mirrors,
        "obstructions": [],
    }


def _mirror(id_, pos, template="sph"):
    return {
        "id": id_,
        "template": template,
        "position": pos,
        "orientation": [0.0, 0.0, 0.0],
        "aperture": {"type": "circular", "radius": 0.5},
    }


def _asphere(curvature=0.05, conic=-1.0):
    return {"type": "aspheric", "curvature": curvature, "conic": conic, "aspheric": []}


def _zernike(coeffs, r_norm=0.5):
    return {"type": "zernike", "coeffs": coeffs, "r_norm": r_norm}


class TestSurfaceListLoad:
    def test_aspheric_plus_zernike_loads_as_sum(self):
        cfg = _config(
            {"sph": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4])]}},
            [_mirror("M_0", [0.0, 0.0, 0.0])],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3, 5e-4])
        assert float(zern.r_norm[0]) == pytest.approx(0.5)

    def test_single_aspheric_stays_bare_asphere(self):
        cfg = _config({"sph": {"surface": _asphere()}}, [_mirror("M_0", [0.0, 0.0, 0.0])])
        tel = build_telescope_config(cfg, 4, KEY)
        assert isinstance(tel.stage(0).surface, AsphericSurfaceGroup)

    def test_standalone_zernike_loads_as_zernike(self):
        cfg = _config(
            {"z": {"surface": _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4])}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="z")],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        assert isinstance(tel.stage(0).surface, ZernikeSurfaceGroup)

    def test_mixed_templates_zero_fill_absent(self):
        # two templates in the same stage+aperture bucket: only one has a zernike
        cfg = _config(
            {
                "zt": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3])]},
                "at": {"surface": _asphere()},
            },
            [
                _mirror("M_0", [0.0, 0.0, 0.0], template="zt"),
                _mirror("M_1", [1.0, 0.0, 0.0], template="at"),
            ],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3])
        assert np.allclose(np.asarray(zern.coeffs[1]), 0.0)

class TestZernikeRoundTrip:
    def test_idempotent_dict_round_trip(self):
        cfg = _config(
            {
                "a": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4])]},
                "b": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 2e-3])]},
            },
            [
                _mirror("M_0", [0.0, 0.0, 0.0], template="a"),
                _mirror("M_1", [1.0, 0.0, 0.0], template="b"),
            ],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        # Two mirrors, each with its own distinct zernike figure error: the
        # zernike term is written directly on each mirror, not on a template.
        assert any(m.get("zernike") is not None for m in d1["mirrors"])

    def test_sag_preserved(self):
        cfg = _config(
            {"a": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4])]}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="a")],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        tel2 = build_telescope_config(telescope_to_dict(tel), 4, KEY)
        s1, s2 = tel.stage(0).surface, tel2.stage(0).surface
        for x, y in [(0.1, 0.05), (-0.2, 0.1), (0.3, -0.25)]:
            assert float(s1.sag_at(0, x, y)) == pytest.approx(float(s2.sag_at(0, x, y)), abs=1e-9)

    def test_perturbed_telescope_round_trips(self):
        cfg = _config(
            {"sph": {"surface": _asphere()}},
            [_mirror("M_0", [0.0, 0.0, 0.0]), _mirror("M_1", [1.0, 0.0, 0.0])],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        tel = ops.apply_astigmatism(tel, 0, 1e-3, jax.random.key(5))
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        # Every facet drew a figure error, written directly on each mirror; the
        # shared aspheric base (untouched by the perturbation) still templatises.
        assert any(m.get("zernike") is not None for m in d1["mirrors"])
        assert len(d1["mirror_templates"]) == 1


def _mirror_group(surface, radii, stage=0):
    n = surface.offsets.shape[0]
    aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=stage,
        n_samples=8,
    )


class TestStandaloneAndGuards:
    def test_standalone_zernike_serializes_as_zernike_surface(self):
        zg = ZernikeSurfaceGroup(
            coeffs=jnp.array([[0.0, 0.0, 0.0, 1e-3, 5e-4]]), r_norm=jnp.array([0.5]),
        )
        tel = Telescope(mirror_groups=[_mirror_group(zg, jnp.array([0.5]))], name="z")
        d = telescope_to_dict(tel)
        # No aspheric base at all -> self-contained, no template; the zernike
        # shape is written directly on the mirror.
        assert "template" not in d["mirrors"][0]
        assert not d.get("mirror_templates")
        zern = d["mirrors"][0]["zernike"]
        assert zern["coeffs"][3] == pytest.approx(1e-3)
        # reloads to a standalone Zernike surface with an equivalent sag
        tel2 = build_telescope_config(d, 4, KEY)
        assert isinstance(tel2.stage(0).surface, ZernikeSurfaceGroup)
        for x, y in [(0.1, 0.05), (-0.2, 0.1)]:
            assert float(tel.stage(0).surface.sag_at(0, x, y)) == pytest.approx(
                float(tel2.stage(0).surface.sag_at(0, x, y)), abs=1e-9
            )

    def test_nonzero_composite_offset_raises(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.05]), conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)), offsets=jnp.zeros((1, 2)),
        )
        zg = ZernikeSurfaceGroup(coeffs=jnp.zeros((1, 4)), r_norm=jnp.array([0.5]))
        bad = SumSurfaceGroup([asph, zg], offsets=jnp.array([[0.1, 0.0]]))
        tel = Telescope(mirror_groups=[_mirror_group(bad, jnp.array([0.5]))], name="z")
        with pytest.raises(ValueError, match="composite decenter"):
            telescope_to_dict(tel)

    def test_zernike_too_many_coeffs_rejected_by_schema(self):
        cfg = _config(
            {"z": {"surface": _zernike([0.0] * 12)}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="z")],
        )
        with pytest.raises(YAMLConfigError):
            build_telescope_config(cfg, 4, KEY)
