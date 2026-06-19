import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from iactrace import Camera, Telescope
from iactrace.io import (
    build_camera_config,
    build_telescope_config,
    save_camera,
    save_telescope,
    telescope_to_dict,
)


@pytest.fixture
def n_samples():
    return 4


@pytest.fixture
def random_key():
    return jax.random.key(0)


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
def simple_camera_config():
    return {
        "camera": {"quantum_efficiency": 1.0},
        "sensors": [
            {
                "id": "sensor_0",
                "type": "square",
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "width": 100,
                "height": 100,
                "bounds": [-1.0, 1.0, -1.0, 1.0],
            }
        ],
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
            "primary": {
                "surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []}
            }
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
            {"id": "cylinder_0", "type": "cylinder", "p1": [0.0, 0.0, 0.0], "p2": [0.0, 0.0, 5.0], "r": 0.1},
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


class TestSaveTelescope:
    def test_save_creates_file(
        self, n_samples, random_key, simple_disk_telescope_config
    ):
        telescope = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            result_path = save_telescope(telescope, filepath)
            assert result_path.exists()
        finally:
            filepath.unlink(missing_ok=True)

    def test_save_overwrite_false_raises(
        self, n_samples, random_key, simple_disk_telescope_config
    ):
        telescope = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            with pytest.raises(FileExistsError):
                save_telescope(telescope, filepath, overwrite=False)
        finally:
            filepath.unlink(missing_ok=True)


# Round-trip


class TestRoundTrip:
    def test_disk_mirror_roundtrip(
        self, n_samples, random_key, simple_disk_telescope_config
    ):
        telescope1 = build_telescope_config(simple_disk_telescope_config, n_samples, random_key)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

            assert telescope1.name == telescope2.name
            assert len(telescope1.mirror_groups) == len(telescope2.mirror_groups)
            for g1, g2 in zip(
                telescope1.mirror_groups, telescope2.mirror_groups, strict=False
            ):
                np.testing.assert_allclose(
                    np.asarray(g1.positions), np.asarray(g2.positions), rtol=1e-5
                )
                np.testing.assert_allclose(
                    np.asarray(g1.rotations), np.asarray(g2.rotations), rtol=1e-5
                )
        finally:
            filepath.unlink(missing_ok=True)

    def test_polygon_mirror_roundtrip(
        self, n_samples, random_key, polygon_telescope_config
    ):
        telescope1 = build_telescope_config(polygon_telescope_config, n_samples, random_key)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

            from iactrace.core.apertures import PolygonAperture
            for g1, g2 in zip(
                telescope1.mirror_groups, telescope2.mirror_groups, strict=False
            ):
                if isinstance(getattr(g1, 'aperture', None), PolygonAperture):
                    assert isinstance(getattr(g2, 'aperture', None), PolygonAperture)
                    np.testing.assert_allclose(
                        np.asarray(g1.aperture.vertices),
                        np.asarray(g2.aperture.vertices),
                        rtol=1e-5,
                    )
        finally:
            filepath.unlink(missing_ok=True)

    def test_obstructions_roundtrip(
        self, n_samples, random_key, telescope_with_obstructions_config
    ):
        telescope1 = build_telescope_config(
            telescope_with_obstructions_config, n_samples, random_key
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

            assert telescope1.obstruction_groups is not None
            assert telescope2.obstruction_groups is not None
            assert len(telescope1.obstruction_groups) == len(telescope2.obstruction_groups)
        finally:
            filepath.unlink(missing_ok=True)

    def test_surface_parameters_preserved(self, n_samples, random_key):
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_telescope(telescope1, filepath, precision=12)
            telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

            for g1, g2 in zip(
                telescope1.mirror_groups, telescope2.mirror_groups, strict=False
            ):
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
        finally:
            filepath.unlink(missing_ok=True)

    def test_hexagonal_sensor_roundtrip(self):
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_camera(camera1, filepath)
            camera2 = Camera.from_yaml(filepath)

            from iactrace.camera.layout import HexagonalSensorGroup
            s1 = camera1.sensor_groups[0]
            s2 = camera2.sensor_groups[0]
            assert isinstance(s1, HexagonalSensorGroup)
            assert isinstance(s2, HexagonalSensorGroup)
            np.testing.assert_allclose(
                np.asarray(s1.hex_centers),
                np.asarray(s2.hex_centers),
                rtol=1e-5,
            )
        finally:
            filepath.unlink(missing_ok=True)

    def test_multiple_stages_preserved(self, n_samples, random_key):
        config = {
            "telescope": {
                "name": "two_stage",
                "units": "m",
                "camera_position": [0.0, 0.0, -1.0],
                "camera_rotation": [0.0, 0.0, 0.0],
            },
            "mirror_templates": {
                "primary": {"surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []}},
                "secondary": {"surface": {"curvature": 0.1, "conic": 0.0, "aspheric": []}},
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, n_samples, key=random_key)

            assert len(telescope2.mirror_groups) == 2
            stages = {g.optical_stage for g in telescope2.mirror_groups}
            assert 0 in stages
            assert 1 in stages
        finally:
            filepath.unlink(missing_ok=True)
    def test_reflectivity_curve_roundtrip(self, n_samples, random_key):
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
                    "surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []},
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)
        try:
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
        finally:
            filepath.unlink(missing_ok=True)

    def test_default_mirror_yaml_unchanged(
        self, n_samples, random_key, simple_disk_telescope_config,
    ):
        """A mirror without curve information saves WITHOUT the new field."""
        telescope = build_telescope_config(
            simple_disk_telescope_config, n_samples, random_key,
        )
        d = telescope_to_dict(telescope)
        # No new fields appear in the round-tripped YAML
        for tpl in d["mirror_templates"].values():
            assert "reflectivity" not in tpl
            assert "coating" not in tpl
        for m in d["mirrors"]:
            assert "reflectivity" not in m

