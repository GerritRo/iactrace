import jax
import jax.numpy as jnp

from iactrace import (
    HexagonalSensorGroup,
    SquareSensorGroup,
    StraightThroughHexagonalSensorGroup,
    StraightThroughSquareSensorGroup,
)
from iactrace.core.intersections import intersect_plane
from iactrace.sensors.hexagonal import (
    SQRT3,
    _detect_hex_grid,
    _find_three_nearest_hexes_and_weights,
)


def make_hex_centers(n_rings=2, hex_size=0.001):
    """Generate hexagonal grid center positions."""
    centers = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(-n_rings, n_rings + 1):
            if max(abs(q), abs(r), abs(-q - r)) <= n_rings:
                x = hex_size * SQRT3 * (q + r / 2)
                y = hex_size * 1.5 * r
                centers.append([x, y])
    return jnp.array(centers)


class TestSquareSensor:
    """Test SquareSensorGroup basic functionality."""

    def test_accumulation_correct_pixel(self):
        """Points accumulate to the correct pixel based on position."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0])
        x = jnp.array([0.1])  # Center of pixel 5
        y = jnp.array([0.1])
        values = jnp.array([1.0])

        result = sensor.accumulate(sensor_idx, x, y, values)

        assert result[0, 5, 5] == 1.0
        assert result.sum() == 1.0

    def test_multiple_hits_same_pixel_sum(self):
        """Multiple hits to the same pixel sum their values."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0, 0, 0])
        x = jnp.array([0.1, 0.15, 0.05])
        y = jnp.array([0.1, 0.1, 0.1])
        values = jnp.array([1.0, 2.0, 3.0])

        result = sensor.accumulate(sensor_idx, x, y, values)

        assert result[0, 5, 5] == 6.0

    def test_outside_bounds_excluded(self):
        """Points outside sensor bounds are excluded."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        sensor_idx = jnp.array([0, 0, 0])
        x = jnp.array([0.0, 5.0, -5.0])  # 5.0 and -5.0 are outside
        y = jnp.array([0.0, 0.0, 0.0])
        values = jnp.array([1.0, 2.0, 3.0])

        result = sensor.accumulate(sensor_idx, x, y, values)

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

        result = sensor.accumulate(sensor_idx, x, y, values)

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

        result = sensor.accumulate(sensor_idx, x, y, values)

        assert result.sum() == 1.0


class TestEdgeExclusion:
    """Test edge exclusion functionality."""

    def test_square_edge_exclusion(self):
        """Points near pixel edges are excluded when edge_width > 0."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
            edge_width=0.05,
        )

        sensor_idx = jnp.array([0])
        values = jnp.array([1.0])

        # Center of pixel should be valid
        result_center = sensor.accumulate(sensor_idx, jnp.array([0.1]), jnp.array([0.1]), values)
        assert result_center.sum() == 1.0

        # Point close to edge should be excluded
        result_edge = sensor.accumulate(sensor_idx, jnp.array([0.01]), jnp.array([0.1]), values)
        assert result_edge.sum() == 0.0


class TestStraightThroughEstimator:
    """Test Straight-Through Estimator (STE) sensors."""

    def test_ste_forward_matches_hard_square(self):
        """STE forward pass matches hard sensor for square sensors."""
        sensor_idx = jnp.array([0, 0, 0, 0])
        x = jnp.array([0.0, 0.005, -0.003, 0.008])
        y = jnp.array([0.0, -0.002, 0.007, 0.001])
        values = jnp.array([1.0, 2.0, 1.5, 0.5])

        hard_sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-0.01, 0.01, -0.01, 0.01),
        )

        ste_sensor = StraightThroughSquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-0.01, 0.01, -0.01, 0.01),
        )

        hard_result = hard_sensor.accumulate(sensor_idx, x, y, values)
        ste_result = ste_sensor.accumulate(sensor_idx, x, y, values)

        assert jnp.allclose(hard_result, ste_result)

    def test_ste_forward_matches_hard_hex(self):
        """STE forward pass matches hard sensor for hexagonal sensors."""
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)

        sensor_idx = jnp.array([0, 0, 0])
        x = jnp.array([0.0, 0.001, -0.002])
        y = jnp.array([0.0, 0.002, -0.001])
        values = jnp.array([1.0, 2.0, 1.5])

        hard_sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        ste_sensor = StraightThroughHexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )

        hard_result = hard_sensor.accumulate(sensor_idx, x, y, values)
        ste_result = ste_sensor.accumulate(sensor_idx, x, y, values)

        assert jnp.allclose(hard_result, ste_result)

    def test_ste_produces_gradients(self):
        """STE backward pass produces non-zero gradients for positions."""
        ste_sensor = StraightThroughSquareSensorGroup(
            positions=[[0.0, 0.0, 1.0]],
            rotations=[[0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-0.01, 0.01, -0.01, 0.01),
        )

        def loss_fn(x, y, values):
            sensor_idx = jnp.zeros(len(x), dtype=jnp.int32)
            image = ste_sensor.accumulate(sensor_idx, x, y, values)
            return jnp.sum(image * jnp.arange(image.size).reshape(image.shape))

        x = jnp.array([0.001, 0.003])
        y = jnp.array([0.002, -0.001])
        values = jnp.array([1.0, 1.0])

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_x, grad_y = grad_fn(x, y, values)

        assert jnp.any(grad_x != 0) or jnp.any(grad_y != 0)


class TestMultiSensor:
    """Test multi-sensor functionality (common for both square and hex)."""

    def test_multi_sensor_isolation_square(self):
        """Rays hitting different sensors accumulate to separate image slices."""
        sensor = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            rotations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )

        assert sensor.n_sensors == 3

        sensor_idx = jnp.array([0, 1, 2])
        x = jnp.array([0.1, 0.1, 0.1])
        y = jnp.array([0.1, 0.1, 0.1])
        values = jnp.array([1.0, 2.0, 3.0])

        result = sensor.accumulate(sensor_idx, x, y, values)

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

        result = sensor.accumulate(sensor_idx, x, y, values)

        assert result.shape == (2, len(hex_centers))
        assert result[0].sum() == 5.0
        assert result[1].sum() == 7.0

    def test_ste_gradient_isolation(self):
        """Gradients only flow from the correct sensor's output to its rays."""
        ste_sensor = StraightThroughSquareSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            rotations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            width=10, height=10,
            bounds=(-0.01, 0.01, -0.01, 0.01),
        )

        def loss_sensor_0_only(x, y, values, sensor_idx):
            image = ste_sensor.accumulate(sensor_idx, x, y, values)
            return jnp.sum(image[0] * jnp.arange(100).reshape(10, 10))

        sensor_idx = jnp.array([0, 1])
        x = jnp.array([0.002, 0.003])
        y = jnp.array([0.001, 0.002])
        values = jnp.array([1.0, 1.0])

        grad_fn = jax.grad(loss_sensor_0_only, argnums=(0, 1))
        grad_x, grad_y = grad_fn(x, y, values, sensor_idx)

        # Only ray 0 should have gradients (hits sensor 0)
        assert grad_x[0] != 0 or grad_y[0] != 0
        assert grad_x[1] == 0 and grad_y[1] == 0


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

        result = sensor.accumulate(sensor_idx, x, y, values)

        assert jnp.sum(result > 0) == len(hex_centers)


class TestBarycentricInterpolation:
    """Test barycentric weight calculation for hex STE."""

    def test_barycentric_weights_sum_to_one(self):
        """Barycentric weights sum to 1 for any point."""
        q_vals = jnp.array([0.3, 0.7, -0.2, 1.5])
        r_vals = jnp.array([0.1, -0.4, 0.8, 0.2])

        q0, r0, w0, q1, r1, w1, q2, r2, w2 = _find_three_nearest_hexes_and_weights(q_vals, r_vals)

        weight_sums = w0 + w1 + w2
        assert jnp.allclose(weight_sums, 1.0, rtol=1e-5)

    def test_barycentric_weights_non_negative(self):
        """Barycentric weights are non-negative."""
        q_vals = jnp.array([0.3, 0.7, -0.2, 1.5, 0.0])
        r_vals = jnp.array([0.1, -0.4, 0.8, 0.2, 0.0])

        q0, r0, w0, q1, r1, w1, q2, r2, w2 = _find_three_nearest_hexes_and_weights(q_vals, r_vals)

        assert jnp.all(w0 >= 0)
        assert jnp.all(w1 >= 0)
        assert jnp.all(w2 >= 0)


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
