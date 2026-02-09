import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest

from iactrace import MCIntegrator, Telescope
from iactrace.io import build_telescope, save_telescope, telescope_to_dict


@pytest.fixture
def integrator():
    return MCIntegrator(n_samples=4)


@pytest.fixture
def random_key():
    return jax.random.key(0)


@pytest.fixture
def simple_disk_config():
    """A simple telescope config with circular mirrors."""
    return {
        "telescope": {"name": "simple_disk", "units": "m"},
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
        "sensors": [
            {
                "id": "sensor_0",
                "type": "square",
                "position": [0.0, 0.0, 10.0],
                "orientation": [0.0, 0.0, 0.0],
                "width": 100,
                "height": 100,
                "bounds": [-1.0, 1.0, -1.0, 1.0],
            }
        ],
        "obstructions": [],
    }


@pytest.fixture
def polygon_config():
    """A telescope config with polygon mirrors."""
    return {
        "telescope": {"name": "polygon_telescope", "units": "m"},
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
        "sensors": [
            {
                "id": "sensor_0",
                "type": "square",
                "position": [0.0, 0.0, 15.0],
                "orientation": [0.0, 0.0, 0.0],
                "width": 50,
                "height": 50,
                "bounds": [-0.5, 0.5, -0.5, 0.5],
            }
        ],
        "obstructions": [],
    }


@pytest.fixture
def config_with_obstructions():
    """A telescope config with various obstruction types."""
    return {
        "telescope": {"name": "with_obstructions", "units": "m"},
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
        "sensors": [
            {
                "id": "sensor_0",
                "type": "square",
                "position": [0.0, 0.0, 10.0],
                "orientation": [0.0, 0.0, 0.0],
                "width": 100,
                "height": 100,
                "bounds": [-1.0, 1.0, -1.0, 1.0],
            }
        ],
        "obstructions": [
            {"id": "cylinder_0", "type": "cylinder", "p1": [0.0, 0.0, 0.0], "p2": [0.0, 0.0, 5.0], "r": 0.1},
            {"id": "box_0", "type": "box", "p1": [-0.5, -0.5, 2.0], "p2": [0.5, 0.5, 2.5]},
            {"id": "sphere_0", "type": "sphere", "center": [1.0, 0.0, 3.0], "r": 0.2},
        ],
    }


class TestTelescopeToDict:
    """Test telescope_to_dict function."""

    def test_conversion_preserves_structure(self, integrator, random_key, simple_disk_config):
        """Conversion preserves mirrors, sensors, and obstructions."""
        telescope = build_telescope(simple_disk_config, integrator, random_key)
        result = telescope_to_dict(telescope)

        assert "telescope" in result
        assert "mirrors" in result
        assert "sensors" in result
        assert result["telescope"]["name"] == "simple_disk"
        assert len(result["mirrors"]) == 2
        assert len(result["sensors"]) == 1


class TestSaveTelescope:
    """Test save_telescope function."""

    def test_save_creates_file(self, integrator, random_key, simple_disk_config):
        """save_telescope creates a file."""
        telescope = build_telescope(simple_disk_config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            result_path = save_telescope(telescope, filepath)
            assert result_path.exists()
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_save_overwrite_false_raises(self, integrator, random_key, simple_disk_config):
        """save raises when file exists and overwrite=False."""
        telescope = build_telescope(simple_disk_config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            with pytest.raises(FileExistsError):
                save_telescope(telescope, filepath, overwrite=False)
        finally:
            if filepath.exists():
                filepath.unlink()


class TestRoundTrip:
    """Test round-trip: load -> save -> load produces equivalent configs."""

    def test_disk_mirror_roundtrip(self, integrator, random_key, simple_disk_config):
        """Round-trip preserves circular mirror data."""
        telescope1 = build_telescope(simple_disk_config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            assert telescope1.name == telescope2.name
            assert len(telescope1.mirror_groups) == len(telescope2.mirror_groups)

            for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
                np.testing.assert_allclose(np.asarray(g1.positions), np.asarray(g2.positions), rtol=1e-5)
                np.testing.assert_allclose(np.asarray(g1.rotations), np.asarray(g2.rotations), rtol=1e-5)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_polygon_mirror_roundtrip(self, integrator, random_key, polygon_config):
        """Round-trip preserves polygon mirror vertices."""
        telescope1 = build_telescope(polygon_config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            from iactrace.telescope.mirrors import AsphericPolygonMirrorGroup

            for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
                if isinstance(g1, AsphericPolygonMirrorGroup):
                    assert isinstance(g2, AsphericPolygonMirrorGroup)
                    np.testing.assert_allclose(np.asarray(g1.vertices), np.asarray(g2.vertices), rtol=1e-5)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_obstructions_roundtrip(self, integrator, random_key, config_with_obstructions):
        """Round-trip preserves obstruction data."""
        telescope1 = build_telescope(config_with_obstructions, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            assert telescope1.obstruction_groups is not None
            assert telescope2.obstruction_groups is not None
            assert len(telescope1.obstruction_groups) == len(telescope2.obstruction_groups)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_surface_parameters_preserved(self, integrator, random_key):
        """Round-trip preserves surface parameters (curvature, conic, aspheric)."""
        config = {
            "telescope": {"name": "aspheric_test", "units": "m"},
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
            "sensors": [
                {
                    "id": "sensor_0",
                    "type": "square",
                    "position": [0.0, 0.0, 10.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1.0, 1.0, -1.0, 1.0],
                }
            ],
            "obstructions": [],
        }

        telescope1 = build_telescope(config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath, precision=12)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            for g1, g2 in zip(telescope1.mirror_groups, telescope2.mirror_groups, strict=False):
                np.testing.assert_allclose(np.asarray(g1.curvatures), np.asarray(g2.curvatures), rtol=1e-10)
                np.testing.assert_allclose(np.asarray(g1.conics), np.asarray(g2.conics), rtol=1e-10)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_hexagonal_sensor_roundtrip(self, integrator, random_key):
        """Round-trip preserves hexagonal sensor centers."""
        config = {
            "telescope": {"name": "hex_sensor_test", "units": "m"},
            "mirror_templates": {"primary": {"surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []}}},
            "mirrors": [
                {
                    "id": "M_0",
                    "template": "primary",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
            "sensors": [
                {
                    "id": "sensor_0",
                    "type": "hexagonal",
                    "position": [0.0, 0.0, 10.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "centers_x": [0.0, 1.0, 2.0, 0.5, 1.5],
                    "centers_y": [0.0, 0.0, 0.0, 0.866, 0.866],
                }
            ],
            "obstructions": [],
        }

        telescope1 = build_telescope(config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            from iactrace.sensors.hexagonal import HexagonalSensorGroup

            s1 = telescope1.sensors[0]
            s2 = telescope2.sensors[0]
            assert isinstance(s1, HexagonalSensorGroup)
            assert isinstance(s2, HexagonalSensorGroup)
            np.testing.assert_allclose(np.asarray(s1.hex_centers), np.asarray(s2.hex_centers), rtol=1e-5)
        finally:
            if filepath.exists():
                filepath.unlink()

    def test_multiple_stages_preserved(self, integrator, random_key):
        """Round-trip preserves optical stages."""
        config = {
            "telescope": {"name": "two_stage", "units": "m"},
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
            "sensors": [
                {
                    "id": "sensor_0",
                    "type": "square",
                    "position": [0.0, 0.0, -1.0],
                    "orientation": [0.0, 0.0, 0.0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-0.5, 0.5, -0.5, 0.5],
                }
            ],
            "obstructions": [],
        }

        telescope1 = build_telescope(config, integrator, random_key)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            filepath = Path(f.name)

        try:
            save_telescope(telescope1, filepath)
            telescope2 = Telescope.from_yaml(filepath, integrator, random_key)

            assert len(telescope2.mirror_groups) == 2
            stages = {g.optical_stage for g in telescope2.mirror_groups}
            assert 0 in stages
            assert 1 in stages
        finally:
            if filepath.exists():
                filepath.unlink()
