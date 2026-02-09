import jax
import pytest

from iactrace import MCIntegrator
from iactrace.io import YAMLConfigError, build_telescope


@pytest.fixture
def integrator():
    return MCIntegrator(n_samples=4)


@pytest.fixture
def random_key():
    return jax.random.key(0)


@pytest.fixture
def valid_template():
    """A valid mirror template for testing."""
    return {
        "test_mirror": {
            "surface": {
                "curvature": 0.1,
                "conic": -1.0,
            }
        }
    }


class TestMirrorConfigErrors:
    """Test error handling for common mirror configuration mistakes."""

    def test_undefined_template_raises(self, integrator, random_key):
        """Reference to undefined template should raise helpful error listing available templates."""
        config = {
            "mirror_templates": {
                "template_a": {"surface": {"curvature": 0.1, "conic": -1}},
                "template_b": {"surface": {"curvature": 0.2, "conic": -1}},
            },
            "mirrors": [
                {
                    "template": "wrong_name",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                }
            ],
            "sensors": [],
        }

        with pytest.raises(YAMLConfigError, match="undefined template 'wrong_name'"):
            build_telescope(config, integrator, random_key)

    def test_missing_required_fields_raises(self, integrator, random_key, valid_template):
        """Missing required fields should raise clear errors."""
        # Missing template
        config_no_template = {
            "mirror_templates": valid_template,
            "mirrors": [{"position": [0, 0, 0], "orientation": [0, 0, 0], "aperture": {"type": "circular", "radius": 0.5}}],
            "sensors": [],
        }
        with pytest.raises(YAMLConfigError, match="missing required 'template' field"):
            build_telescope(config_no_template, integrator, random_key)

        # Missing aperture
        config_no_aperture = {
            "mirror_templates": valid_template,
            "mirrors": [{"template": "test_mirror", "position": [0, 0, 0], "orientation": [0, 0, 0]}],
            "sensors": [],
        }
        with pytest.raises(YAMLConfigError, match="missing required 'aperture' field"):
            build_telescope(config_no_aperture, integrator, random_key)

    def test_invalid_aperture_type_raises(self, integrator, random_key, valid_template):
        """Unknown aperture type should raise YAMLConfigError."""
        config = {
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "elliptical"},  # Invalid type
                }
            ],
            "sensors": [],
        }

        with pytest.raises(YAMLConfigError, match="unknown aperture type 'elliptical'"):
            build_telescope(config, integrator, random_key)


class TestSensorConfigErrors:
    """Test error handling for sensor configuration issues."""

    def test_invalid_sensor_type_raises(self, integrator, random_key):
        """Unknown sensor type should raise YAMLConfigError."""
        config = {
            "mirrors": [],
            "sensors": [
                {
                    "type": "triangular",  # Invalid
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                }
            ],
        }

        with pytest.raises(YAMLConfigError, match="unknown type 'triangular'"):
            build_telescope(config, integrator, random_key)

    def test_square_sensor_wrong_bounds_raises(self, integrator, random_key):
        """Square sensor with wrong bounds count should raise YAMLConfigError."""
        config = {
            "mirrors": [],
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1, 1, -1],  # Only 3 values, need 4
                }
            ],
        }

        with pytest.raises(YAMLConfigError, match="bounds must have 4 values"):
            build_telescope(config, integrator, random_key)

    def test_hexagonal_sensor_mismatched_centers_raises(self, integrator, random_key):
        """Hexagonal sensor with mismatched center arrays should raise YAMLConfigError."""
        config = {
            "mirrors": [],
            "sensors": [
                {
                    "type": "hexagonal",
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                    "centers_x": [0, 1, 2],
                    "centers_y": [0, 1],  # Different length
                }
            ],
        }

        with pytest.raises(YAMLConfigError, match="must have same length"):
            build_telescope(config, integrator, random_key)


class TestValidConfigs:
    """Test that valid configurations load correctly."""

    def test_valid_disk_mirror_loads(self, integrator, random_key, valid_template):
        """Valid circular mirror configuration should load successfully."""
        config = {
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                }
            ],
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1, 1, -1, 1],
                }
            ],
        }

        telescope = build_telescope(config, integrator, random_key)
        assert len(telescope.mirror_groups) == 1
        assert len(telescope.sensors) == 1

    def test_valid_polygon_mirror_loads(self, integrator, random_key, valid_template):
        """Valid polygon mirror configuration should load successfully."""
        config = {
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {
                        "type": "polygon",
                        "vertices": [[0, 0], [1, 0], [0.5, 1]],
                    },
                }
            ],
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1, 1, -1, 1],
                }
            ],
        }

        telescope = build_telescope(config, integrator, random_key)
        assert len(telescope.mirror_groups) == 1

    def test_empty_config_loads(self, integrator, random_key):
        """Empty configuration should load without errors."""
        config = {
            "mirrors": [],
            "sensors": [],
        }

        telescope = build_telescope(config, integrator, random_key)
        assert len(telescope.mirror_groups) == 0
        assert len(telescope.sensors) == 0

    def test_per_mirror_surface_overrides(self, integrator, random_key, valid_template):
        """Mirrors can override template surface parameters."""
        config = {
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                    "curvature": 0.2,  # Override template's 0.1
                },
                {
                    "template": "test_mirror",
                    "position": [1, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                    # No overrides - uses template values
                },
            ],
            "sensors": [],
        }

        telescope = build_telescope(config, integrator, random_key)
        group = telescope.mirror_groups[0]

        # Check per-mirror curvatures
        assert float(group.curvatures[0]) == pytest.approx(0.2)
        assert float(group.curvatures[1]) == pytest.approx(0.1)  # template default
