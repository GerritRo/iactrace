import jax.numpy as jnp
import pytest

from iactrace import ConstantQE, HexagonalSensorGroup, SquareSensorGroup
from iactrace.camera._hexgeom import _detect_hex_grid
from iactrace.core.intersections import intersect_plane

from ._helpers import bin_positions, make_hex_centers


def _square(**kwargs):
    base = dict(
        positions=[[0.0, 0.0, 1.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=10,
        height=10,
        bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    base.update(kwargs)
    return SquareSensorGroup(**base)


class TestSquareSensor:
    """SquareSensorGroup binning."""

    def test_accumulation_and_summing(self):
        """Hits land in the right pixel; multiple hits to one pixel sum."""
        sensor = _square()
        # single hit -> pixel (5, 5)
        single = bin_positions(sensor, jnp.array([0]), jnp.array([0.1]), jnp.array([0.1]), jnp.array([1.0]))
        assert single[0, 5, 5] == 1.0
        assert single.sum() == 1.0
        # three hits to the same pixel sum their values
        summed = bin_positions(
            sensor,
            jnp.array([0, 0, 0]),
            jnp.array([0.1, 0.15, 0.05]),
            jnp.array([0.1, 0.1, 0.1]),
            jnp.array([1.0, 2.0, 3.0]),
        )
        assert summed[0, 5, 5] == 6.0

    def test_outside_bounds_excluded(self):
        """Points outside sensor bounds are excluded."""
        sensor = _square()
        result = bin_positions(
            sensor,
            jnp.array([0, 0, 0]),
            jnp.array([0.0, 5.0, -5.0]),  # 5.0 and -5.0 are outside
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([1.0, 2.0, 3.0]),
        )
        assert result.sum() == 1.0


class TestHexagonalSensor:
    """HexagonalSensorGroup binning."""

    def test_accumulation_correct_pixel(self):
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]], rotations=[[0.0, 0.0, 0.0]], hex_centers=hex_centers
        )
        result = bin_positions(sensor, jnp.array([0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]))
        assert result.sum() == 1.0
        assert jnp.max(result) == 1.0

    def test_outside_grid_excluded(self):
        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]], rotations=[[0.0, 0.0, 0.0]], hex_centers=hex_centers
        )
        result = bin_positions(
            sensor,
            jnp.array([0, 0]),
            jnp.array([0.0, 1.0]),  # 1.0 is way outside
            jnp.array([0.0, 0.0]),
            jnp.array([1.0, 2.0]),
        )
        assert result.sum() == 1.0


class TestEdgeExclusion:
    def test_square_edge_exclusion(self):
        """Points near pixel edges are excluded when edge_width > 0."""
        sensor = _square(edge_width=0.05)
        # centre of a pixel is valid
        center = bin_positions(sensor, jnp.array([0]), jnp.array([0.1]), jnp.array([0.1]), jnp.array([1.0]))
        assert center.sum() == 1.0
        # a point near the pixel edge is excluded
        edge = bin_positions(sensor, jnp.array([0]), jnp.array([0.01]), jnp.array([0.1]), jnp.array([1.0]))
        assert edge.sum() == 0.0


class TestMultiSensor:
    def test_multi_sensor_isolation(self):
        """Rays hitting different sensors accumulate to separate image slices
        (checked for both square and hex layouts)."""
        square = SquareSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            rotations=[[0.0, 0.0, 0.0]] * 3,
            width=10,
            height=10,
            bounds=(-1.0, 1.0, -1.0, 1.0),
        )
        assert square.n_sensors == 3
        res = bin_positions(
            square, jnp.array([0, 1, 2]), jnp.array([0.1, 0.1, 0.1]), jnp.array([0.1, 0.1, 0.1]),
            jnp.array([1.0, 2.0, 3.0]),
        )
        assert res.shape == (3, 10, 10)
        assert res[0].sum() == 1.0 and res[1].sum() == 2.0 and res[2].sum() == 3.0

        hex_centers = make_hex_centers(n_rings=2, hex_size=0.002)
        hexs = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            rotations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            hex_centers=hex_centers,
        )
        res_h = bin_positions(
            hexs, jnp.array([0, 1]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array([5.0, 7.0])
        )
        assert res_h.shape == (2, len(hex_centers))
        assert res_h[0].sum() == 5.0 and res_h[1].sum() == 7.0


class TestHexGridInfrastructure:
    """Hexagonal grid detection and lookup."""

    def test_detect_hex_grid_correct_size(self):
        hex_size = 0.005
        detected_size, _rotation, _offset = _detect_hex_grid(make_hex_centers(n_rings=3, hex_size=hex_size))
        assert jnp.isclose(detected_size, hex_size, rtol=1e-5)

    def test_lookup_table_all_centers_valid(self):
        """Every hex center maps to a valid index in the lookup table."""
        hex_centers = make_hex_centers(n_rings=3, hex_size=0.002)
        sensor = HexagonalSensorGroup(
            positions=[[0.0, 0.0, 1.0]], rotations=[[0.0, 0.0, 0.0]], hex_centers=hex_centers
        )
        result = bin_positions(
            sensor,
            jnp.zeros(len(hex_centers), dtype=jnp.int32),
            hex_centers[:, 0],
            hex_centers[:, 1],
            jnp.ones(len(hex_centers)),
        )
        assert jnp.sum(result > 0) == len(hex_centers)


class TestPlaneIntersection:
    """Ray-plane intersection (used for sensor hit detection)."""

    def test_ray_hits_horizontal_plane(self):
        xy, t = intersect_plane(
            jnp.array([1.0, 2.0, 10.0]), jnp.array([0.0, 0.0, -1.0]),
            jnp.array([0.0, 0.0, 0.0]), jnp.eye(3),
        )
        assert jnp.allclose(xy, jnp.array([1.0, 2.0]), atol=1e-10)
        assert jnp.allclose(t, 10.0, atol=1e-10)

    def test_parallel_or_receding_ray_misses(self):
        """A ray parallel to the plane, or receding from it, returns a sentinel."""
        for direction in (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])):
            _, t = intersect_plane(
                jnp.array([0.0, 0.0, 5.0]), direction, jnp.array([0.0, 0.0, 0.0]), jnp.eye(3)
            )
            assert jnp.isinf(t)


class TestValidation:
    """Constructor input validation across sensor groups, photodetector, camera."""

    def test_square_rejects_bad_geometry(self):
        """Bad positions shape, mismatched N, non-positive dims, degenerate
        bounds and negative edge_width are each rejected."""
        with pytest.raises(ValueError, match="positions"):
            _square(positions=[[0.0, 0.0]])  # (1, 2), not (1, 3)
        with pytest.raises(ValueError, match="same N"):
            _square(positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 2 positions, 1 rotation
        with pytest.raises(ValueError, match="width and height"):
            _square(width=0)
        with pytest.raises(ValueError, match="bounds"):
            _square(bounds=(1.0, 1.0, -1.0, 1.0))  # degenerate x-range
        with pytest.raises(ValueError, match="edge_width"):
            _square(edge_width=-0.1)

    def test_hex_rejects_bad_centers(self):
        with pytest.raises(ValueError, match="hex_centers"):
            HexagonalSensorGroup([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], jnp.zeros((0, 2)))  # empty
        with pytest.raises(ValueError, match="hex_centers"):
            HexagonalSensorGroup([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])  # (1, 3)

    def test_constant_qe_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="qe"):
            ConstantQE(1.5)
        with pytest.raises(ValueError, match="qe"):
            ConstantQE(-0.1)
