import jax
import pytest

from iactrace.io import (
    YAMLConfigError,
    build_camera_config,
    build_telescope_config,
)


@pytest.fixture
def n_samples():
    return 4


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


# ---------------------------------------------------------------------------
# Telescope schema errors
# ---------------------------------------------------------------------------


class TestMirrorConfigErrors:
    """Test error handling for common mirror configuration mistakes."""

    def test_undefined_template_raises(self, n_samples, random_key):
        """Reference to undefined template should raise helpful error."""
        config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
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
        }
        with pytest.raises(YAMLConfigError, match="undefined template 'wrong_name'"):
            build_telescope_config(config, n_samples, random_key)

    def test_missing_required_fields_raises(self, n_samples, random_key, valid_template):
        """Missing required fields should raise clear errors."""
        # Missing template
        config_no_template = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="Field required"):
            build_telescope_config(config_no_template, n_samples, random_key)

        # Missing aperture
        config_no_aperture = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="Field required"):
            build_telescope_config(config_no_aperture, n_samples, random_key)

    def test_invalid_aperture_type_raises(self, n_samples, random_key, valid_template):
        """Unknown aperture type should raise YAMLConfigError."""
        config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "elliptical"},  # Invalid type
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="does not match any of the expected tags"):
            build_telescope_config(config, n_samples, random_key)

    def test_missing_camera_position_raises(self, n_samples, random_key):
        """Telescope config without camera_position should raise YAMLConfigError."""
        config = {
            "telescope": {"name": "no_cam_pos"},
            "mirrors": [],
        }
        with pytest.raises(YAMLConfigError, match="Field required"):
            build_telescope_config(config, n_samples, random_key)

    def test_telescope_rejects_sensors_section(self, n_samples, random_key):
        """Telescope schema is strict — combined-format sensors are forbidden."""
        config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirrors": [],
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 5],
                    "orientation": [0, 0, 0],
                    "width": 10,
                    "height": 10,
                    "bounds": [-1, 1, -1, 1],
                }
            ],
        }
        with pytest.raises(YAMLConfigError):
            build_telescope_config(config, n_samples, random_key)

    def test_telescope_rejects_camera_section(self, n_samples, random_key):
        """Telescope schema is strict — combined-format camera block is forbidden."""
        config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirrors": [],
            "camera": {"quantum_efficiency": 0.5},
        }
        with pytest.raises(YAMLConfigError):
            build_telescope_config(config, n_samples, random_key)


# ---------------------------------------------------------------------------
# Camera schema errors
# ---------------------------------------------------------------------------


class TestSensorConfigErrors:
    """Test error handling for sensor configuration issues."""

    def test_invalid_sensor_type_raises(self):
        """Unknown sensor type should raise YAMLConfigError."""
        config = {
            "sensors": [
                {
                    "type": "triangular",  # Invalid
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="does not match any of the expected tags"):
            build_camera_config(config)

    def test_square_sensor_wrong_bounds_raises(self):
        """Square sensor with wrong bounds count should raise YAMLConfigError."""
        config = {
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1, 1, -1],  # Only 3 values, need 4
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="at least 4 items"):
            build_camera_config(config)

    def test_hexagonal_sensor_mismatched_centers_raises(self):
        """Hexagonal sensor with mismatched center arrays should raise."""
        config = {
            "sensors": [
                {
                    "type": "hexagonal",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "centers_x": [0, 1, 2],
                    "centers_y": [0, 1],  # Different length
                }
            ],
        }
        with pytest.raises(YAMLConfigError, match="must have same length"):
            build_camera_config(config)

    def test_camera_rejects_extra_top_level_keys(self):
        """Stray keys (e.g. 'frame: world') are hard errors."""
        config = {
            "sensors": [],
            "frame": "world",
        }
        with pytest.raises(YAMLConfigError):
            build_camera_config(config)


# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


class TestValidConfigs:
    """Test that valid configurations load correctly."""

    def test_valid_disk_mirror_loads(self, n_samples, random_key, valid_template):
        """Valid circular mirror configuration should load successfully."""
        tel_config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": valid_template,
            "mirrors": [
                {
                    "template": "test_mirror",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                }
            ],
        }
        cam_config = {
            "sensors": [
                {
                    "type": "square",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "width": 100,
                    "height": 100,
                    "bounds": [-1, 1, -1, 1],
                }
            ],
        }

        telescope = build_telescope_config(tel_config, n_samples, random_key)
        camera = build_camera_config(cam_config)
        assert len(telescope.mirror_groups) == 1
        assert len(camera.sensor_groups) == 1

    def test_valid_polygon_mirror_loads(self, n_samples, random_key, valid_template):
        """Valid polygon mirror configuration should load successfully."""
        tel_config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
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
        }
        telescope = build_telescope_config(tel_config, n_samples, random_key)
        assert len(telescope.mirror_groups) == 1

    def test_empty_telescope_loads(self, n_samples, random_key):
        """Empty telescope configuration should load without errors."""
        tel_config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
            "mirrors": [],
        }
        telescope = build_telescope_config(tel_config, n_samples, random_key)
        assert len(telescope.mirror_groups) == 0

    def test_empty_camera_loads(self):
        """Empty camera configuration should load without errors."""
        camera = build_camera_config({"sensors": []})
        assert len(camera.sensor_groups) == 0

    def test_per_mirror_surface_overrides(self, n_samples, random_key, valid_template):
        """Mirrors can override template surface parameters."""
        config = {
            "telescope": {"camera_position": [0, 0, 0], "camera_rotation": [0, 0, 0]},
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
        }
        telescope = build_telescope_config(config, n_samples, random_key)
        group = telescope.mirror_groups[0]

        # Check per-mirror curvatures
        assert float(group.surface.curvatures[0]) == pytest.approx(0.2)
        assert float(group.surface.curvatures[1]) == pytest.approx(0.1)  # template default
class TestCoatingsFromYaml:
    """Inline R(θ) / T(θ) curves loaded from YAML."""

    def test_tabulated_reflectivity_loads(self, n_samples, random_key):
        from iactrace.core.coatings import TabulatedCoating

        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": {
                "protected_silver": {
                    "surface": {"curvature": 0.1, "conic": -1.0},
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 30.0, 60.0, 80.0],
                        "values": [0.96, 0.95, 0.90, 0.60],
                    },
                },
            },
            "mirrors": [
                {
                    "template": "protected_silver",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
        }
        tel = build_telescope_config(config, n_samples, random_key)
        interaction = tel.mirror_groups[0].interaction_module
        assert isinstance(interaction.reflectivity, TabulatedCoating)
        # Scalar defaults to 1.0 when no per-mirror override
        assert float(interaction.reflectivity_scalar[0]) == pytest.approx(1.0)

    def test_scalar_reflectivity_on_template(self, n_samples, random_key):
        """A scalar template reflectivity becomes the bulk multiplier, with
        no coating attached."""
        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": {
                "matte": {
                    "surface": {"curvature": 0.1, "conic": -1.0},
                    "reflectivity": 0.85,
                },
            },
            "mirrors": [
                {
                    "template": "matte",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
        }
        tel = build_telescope_config(config, n_samples, random_key)
        interaction = tel.mirror_groups[0].interaction_module
        assert interaction.reflectivity is None
        assert float(interaction.reflectivity_scalar[0]) == pytest.approx(0.85)

    def test_per_mirror_reflectivity_override(self, n_samples, random_key):
        """Mirror-level scalar overrides the template's bulk value."""
        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": {
                "tpl": {"surface": {"curvature": 0.1, "conic": -1.0}},
            },
            "mirrors": [
                {
                    "template": "tpl",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                    "reflectivity": 0.7,
                },
                {
                    "template": "tpl",
                    "position": [1, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
        }
        tel = build_telescope_config(config, n_samples, random_key)
        scalars = tel.mirror_groups[0].interaction_module.reflectivity_scalar
        assert float(scalars[0]) == pytest.approx(0.7)
        assert float(scalars[1]) == pytest.approx(1.0)

    def test_mismatched_curves_in_same_bucket_raise(self, n_samples, random_key):
        """Different curves on mirrors that would land in the same group fail."""
        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": {
                "tpl_a": {
                    "surface": {"curvature": 0.1, "conic": -1.0},
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 90.0],
                        "values": [0.9, 0.1],
                    },
                },
                "tpl_b": {
                    "surface": {"curvature": 0.2, "conic": -1.0},
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 90.0],
                        "values": [0.8, 0.2],
                    },
                },
            },
            "mirrors": [
                {
                    "template": "tpl_a",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
                {
                    "template": "tpl_b",
                    "position": [1, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
        }
        with pytest.raises(YAMLConfigError, match="single coating"):
            build_telescope_config(config, n_samples, random_key)

    def test_tabulated_transmittance_on_lens(self, n_samples, random_key):
        from iactrace.core.coatings import TabulatedCoating

        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "lenses": [
                {
                    "type": "aspheric_disk",
                    "position": [0, 0, 0.1],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.05},
                    "curvature": 5.0,
                    "conic": 0.0,
                    "n_inside": 1.5,
                    "stage": 1,
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 60.0, 89.0],
                        "values": [0.98, 0.95, 0.10],
                    },
                },
            ],
        }
        tel = build_telescope_config(config, n_samples, random_key)
        interaction = tel.lens_groups[0].interaction_module
        assert isinstance(interaction.transmittance, TabulatedCoating)

    def test_lens_with_no_coating_defaults_to_fresnel(self, n_samples, random_key):
        """A lens with no ``coating`` field falls back to bare-interface Fresnel,
        which is represented internally as ``transmittance=None``.
        """
        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "lenses": [
                {
                    "type": "aspheric_disk",
                    "position": [0, 0, 0.1],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.05},
                    "curvature": 5.0,
                    "conic": 0.0,
                    "n_inside": 1.5,
                    "stage": 1,
                },
            ],
        }
        tel = build_telescope_config(config, n_samples, random_key)
        interaction = tel.lens_groups[0].interaction_module
        assert interaction.transmittance is None

    def test_invalid_curve_value_rejected(self, n_samples, random_key):
        """Values outside [0, 1] should raise."""
        config = {
            "telescope": {"camera_position": [0, 0, 5], "camera_rotation": [0, 0, 0]},
            "mirror_templates": {
                "bad": {
                    "surface": {"curvature": 0.1, "conic": -1.0},
                    "coating": {
                        "type": "table",
                        "angles_deg": [0.0, 45.0],
                        "values": [1.5, 0.1],  # 1.5 > 1.0
                    },
                },
            },
            "mirrors": [
                {
                    "template": "bad",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "aperture": {"type": "circular", "radius": 0.5},
                },
            ],
        }
        with pytest.raises(YAMLConfigError):
            build_telescope_config(config, n_samples, random_key)
