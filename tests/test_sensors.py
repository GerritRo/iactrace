import jax.numpy as jnp
import pytest

from iactrace import ConstantQE, HexagonalSensorGroup, SquareSensorGroup
from iactrace.camera._hexgeom import _detect_hex_grid
from iactrace.core.intersections import intersect_plane

from ._helpers import bin_positions, make_hex_centers


class TestSquareSensor:
    """Test SquareSensorGroup basic functionality."""

    def test_accumulation_correct_pixel(self):
        """Points accumulate to the correct pixel based on position."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0])
        x = jnp.array([0.1])  # Center of pixel 5
        y = jnp.array([0.1])
        values = jnp.array([1.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result[0, 5, 5] == 1.0
        assert result.sum() == 1.0

    def test_multiple_hits_same_pixel_sum(self):
        """Multiple hits to the same pixel sum their values."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0, 0, 0])
        x = jnp.array([0.1, 0.15, 0.05])
        y = jnp.array([0.1, 0.1, 0.1])
        values = jnp.array([1.0, 2.0, 3.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result[0, 5, 5] == 6.0

    def test_outside_bounds_excluded(self):
        """Points outside sensor bounds are excluded."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0, 0, 0])
        x = jnp.array([0.0, 5.0, -5.0])  # 5.0 and -5.0 are outside
        y = jnp.array([0.0, 0.0, 0.0])
        values = jnp.array([1.0, 2.0, 3.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result.sum() == 1.0


class TestHexagonalSensor:
    """Test HexagonalSensorGroup basic functionality."""

    def test_accumulation_correct_pixel(self):
        """Points accumulate to the correct hexagonal pixel."""
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)

        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        sensor_idx = jnp.array([0])
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        values = jnp.array([1.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result.sum() == 1.0
        assert jnp.max(result) == 1.0

    def test_outside_grid_excluded(self):
        """Points outside the hex grid are excluded."""
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)

        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        sensor_idx = jnp.array([0, 0])
        x = jnp.array([0.0, 1.0])  # 1.0 is way outside
        y = jnp.array([0.0, 0.0])
        values = jnp.array([1.0, 2.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result.sum() == 1.0


class TestEdgeExclusion:
    """Test edge exclusion functionality."""

    def test_square_edge_exclusion(self):
        """Points near pixel edges are excluded when edge_width > 0."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
            edge_width=0.05,
        )

        sensor_idx = jnp.array([0])
        values = jnp.array([1.0])

        # Center of pixel should be valid
        result_center = bin_positions(
            sensor, sensor_idx, jnp.array([0.1]), jnp.array([0.1]), values
        )
        assert result_center.sum() == 1.0

        # Point close to edge should be excluded
        result_edge = bin_positions(sensor, sensor_idx, jnp.array([0.01]), jnp.array([0.1]), values)
        assert result_edge.sum() == 0.0


class TestMultiSensor:
    """Test multi-sensor functionality (common for both square and hex)."""

    def test_multi_sensor_isolation_square(self):
        """Rays hitting different sensors accumulate to separate image slices."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            rotations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        assert sensor.n_sensors == 3

        sensor_idx = jnp.array([0, 1, 2])
        x = jnp.array([0.1, 0.1, 0.1])
        y = jnp.array([0.1, 0.1, 0.1])
        values = jnp.array([1.0, 2.0, 3.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result.shape == (3, 10, 10)
        assert result[0].sum() == 1.0
        assert result[1].sum() == 2.0
        assert result[2].sum() == 3.0

    def test_multi_sensor_isolation_hex(self):
        """Rays hitting different hex sensors accumulate separately."""
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)

        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            rotations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        assert sensor.n_sensors == 2

        sensor_idx = jnp.array([0, 1])
        x = jnp.array([0.0, 0.0])
        y = jnp.array([0.0, 0.0])
        values = jnp.array([5.0, 7.0])

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert result.shape == (2, len(hex_centers))
        assert result[0].sum() == 5.0
        assert result[1].sum() == 7.0


class TestHexGridInfrastructure:
    """Test hexagonal grid detection and lookup."""

    def test_detect_hex_grid_correct_size(self):
        """Grid detection correctly identifies hex size."""
        hex_size = 0.005
        hex_centers = make_hex_centers(n_rings=3, hex_size=hex_size)

        detected_size, rotation, offset = _detect_hex_grid(hex_centers)

        assert jnp.isclose(detected_size, hex_size, rtol=1e-5)

    def test_lookup_table_all_centers_valid(self):
        """Every hex center maps to a valid index in the lookup table."""
        hex_centers = make_hex_centers(n_rings=3, hex_size=0.002)

        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        sensor_idx = jnp.zeros(len(hex_centers), dtype=jnp.int32)
        x = hex_centers[:, 0]
        y = hex_centers[:, 1]
        values = jnp.ones(len(hex_centers))

        result = bin_positions(sensor, sensor_idx, x, y, values)

        assert jnp.sum(result > 0) == len(hex_centers)


class TestPlaneIntersection:
    """Test ray-plane intersection (used for sensor hit detection)."""

    def test_ray_hits_horizontal_plane(self):
        """Ray going down hits horizontal plane at known point."""
        ray_origin = jnp.array([1.0, 2.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])
        plane_center = jnp.array([0.0, 0.0, 0.0])
        plane_rotation = jnp.eye(3)

        xy, t = intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation)

        assert jnp.allclose(xy, jnp.array([1.0, 2.0]), atol=1e-10)
        assert jnp.allclose(t, 10.0, atol=1e-10)

    def test_ray_parallel_to_plane_no_hit(self):
        """Ray parallel to plane returns sentinel value."""
        ray_origin = jnp.array([0.0, 0.0, 5.0])
        ray_direction = jnp.array([1.0, 0.0, 0.0])
        plane_center = jnp.array([0.0, 0.0, 0.0])
        plane_rotation = jnp.eye(3)

        xy, t = intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation)

        assert jnp.isinf(t)

    def test_ray_behind_plane_no_hit(self):
        """Ray going away from plane returns sentinel value."""
        ray_origin = jnp.array([0.0, 0.0, 5.0])
        ray_direction = jnp.array([0.0, 0.0, 1.0])
        plane_center = jnp.array([0.0, 0.0, 0.0])
        plane_rotation = jnp.eye(3)

        xy, t = intersect_plane(ray_origin, ray_direction, plane_center, plane_rotation)

        assert jnp.isinf(t)


class TestValidation:
    """Constructor input validation across sensor groups, photodetector, camera."""

    def test_square_rejects_bad_positions_shape(self):
        with pytest.raises(ValueError, match="positions"):
            SquareSensorGroup(
                positions=[[0.0, 0.0]],  # (1, 2), not (1, 3)
                rotations=[[0.0, 0.0, 0.0]],
                width=4,
                height=4,
                bounds=(-1.0, 1.0, -1.0, 1.0),
            )

    def test_square_rejects_mismatched_n(self):
        with pytest.raises(ValueError, match="same N"):
            SquareSensorGroup(
                positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                width=4,
                height=4,
                bounds=(-1.0, 1.0, -1.0, 1.0),
            )

    def test_square_rejects_nonpositive_dims(self):
        with pytest.raises(ValueError, match="width and height"):
            SquareSensorGroup(
                positions=[[0.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                width=0,
                height=4,
                bounds=(-1.0, 1.0, -1.0, 1.0),
            )

    def test_square_rejects_degenerate_bounds(self):
        with pytest.raises(ValueError, match="bounds"):
            SquareSensorGroup(
                positions=[[0.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                width=4,
                height=4,
                bounds=(1.0, 1.0, -1.0, 1.0),
            )

    def test_square_rejects_negative_edge_width(self):
        with pytest.raises(ValueError, match="edge_width"):
            SquareSensorGroup(
                positions=[[0.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                width=4,
                height=4,
                bounds=(-1.0, 1.0, -1.0, 1.0),
                edge_width=-0.1,
            )

    def test_hex_rejects_empty_centers(self):
        with pytest.raises(ValueError, match="hex_centers"):
            HexagonalSensorGroup(
                positions=[[0.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                hex_centers=jnp.zeros((0, 2)),
            )

    def test_hex_rejects_bad_centers_shape(self):
        with pytest.raises(ValueError, match="hex_centers"):
            HexagonalSensorGroup(
                positions=[[0.0, 0.0, 0.0]],
                rotations=[[0.0, 0.0, 0.0]],
                hex_centers=[[0.0, 0.0, 0.0]],  # (1, 3), not (M, 2)
            )

    def test_constant_qe_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="qe"):
            ConstantQE(1.5)
        with pytest.raises(ValueError, match="qe"):
            ConstantQE(-0.1)
