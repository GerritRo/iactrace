import jax.numpy as jnp

from iactrace.core.obstructions import (
    BoxGroup,
    CylinderGroup,
    OpenCylinderGroup,
    OrientedBoxGroup,
    SphereGroup,
    TriangleGroup,
)


class TestCylinderGroup:
    """Test cylinder obstruction group."""

    def test_ray_hits_cylinder(self):
        """Ray hitting cylinder returns valid t."""
        cylinder = CylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 10]],
            r=[1.0],
        )

        ray_origin = jnp.array([5.0, 0.0, 5.0])
        ray_direction = jnp.array([-1.0, 0.0, 0.0])

        t = cylinder.intersect(ray_origin, ray_direction)

        # Should hit at t=4 (5 - 1 radius)
        assert jnp.isclose(t, 4.0, atol=1e-6)


class TestOpenCylinderGroup:
    """Test open cylinder (no caps) obstruction group."""

    def test_ray_through_cap_misses(self):
        """Ray through cap area misses open cylinder."""
        open_cyl = OpenCylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 10]],
            r=[1.0],
        )

        closed_cyl = CylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 10]],
            r=[1.0],
        )

        ray_origin = jnp.array([0.0, 0.0, -5.0])
        ray_direction = jnp.array([0.0, 0.0, 1.0])

        t_open = open_cyl.intersect(ray_origin, ray_direction)
        t_closed = closed_cyl.intersect(ray_origin, ray_direction)

        # Open cylinder misses (no cap)
        assert jnp.isinf(t_open)
        # Closed cylinder hits cap at z=0
        assert jnp.isclose(t_closed, 5.0, atol=1e-6)

    def test_ray_hits_curved_surface(self):
        """Ray hitting curved surface returns valid t."""
        open_cyl = OpenCylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 10]],
            r=[1.0],
        )

        ray_origin = jnp.array([5.0, 0.0, 5.0])
        ray_direction = jnp.array([-1.0, 0.0, 0.0])

        t = open_cyl.intersect(ray_origin, ray_direction)

        assert jnp.isclose(t, 4.0, atol=1e-6)


class TestBoxGroup:
    """Test axis-aligned box obstruction group."""

    def test_ray_hits_and_misses_box(self):
        """A ray through the box hits its top face; one beside it misses."""
        box = BoxGroup(p1=[[0, 0, 0]], p2=[[2, 2, 2]])
        # through the middle -> hits top face at z=2, so t=8
        t_hit = box.intersect(jnp.array([1.0, 1.0, 10.0]), jnp.array([0.0, 0.0, -1.0]))
        assert jnp.isclose(t_hit, 8.0, atol=1e-6)
        # beside the box -> misses
        t_miss = box.intersect(jnp.array([5.0, 5.0, 10.0]), jnp.array([0.0, 0.0, -1.0]))
        assert jnp.isinf(t_miss)


class TestSphereGroup:
    """Test sphere obstruction group."""

    def test_ray_hits_sphere(self):
        """Ray hitting sphere returns valid t."""
        sphere = SphereGroup(
            centers=[[0, 0, 0]],
            radii=[2.0],
        )

        ray_origin = jnp.array([0.0, 0.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = sphere.intersect(ray_origin, ray_direction)

        # Hits at z=2 (radius from center), so t=8
        assert jnp.isclose(t, 8.0, atol=1e-6)


class TestTriangleGroup:
    """Test triangle obstruction group."""

    def test_ray_hits_triangle(self):
        """Ray hitting triangle returns valid t."""
        triangles = TriangleGroup(
            v0=[[0, 0, 0]],
            v1=[[2, 0, 0]],
            v2=[[1, 2, 0]],
        )

        # Ray through centroid
        centroid = jnp.array([1.0, 2.0 / 3.0, 0.0])
        ray_origin = centroid + jnp.array([0.0, 0.0, 5.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = triangles.intersect(ray_origin, ray_direction)

        assert jnp.isclose(t, 5.0, atol=1e-6)


class TestOrientedBoxGroup:
    """Test oriented box obstruction group."""

    def test_ray_hits_oriented_box(self):
        """Ray hitting oriented box returns valid t."""
        box = OrientedBoxGroup(
            centers=[[0, 0, 0]],
            half_extents=[[1, 1, 1]],
            rotations=[jnp.eye(3)],
        )

        ray_origin = jnp.array([0.0, 0.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = box.intersect(ray_origin, ray_direction)

        # Hits at z=1 (top of box), so t=9
        assert jnp.isclose(t, 9.0, atol=1e-6)

    def test_rotated_box_changes_intersection(self):
        """Box rotated 45 degrees has different intersection distance."""
        angle = jnp.pi / 4
        rotation = jnp.array(
            [
                [jnp.cos(angle), -jnp.sin(angle), 0.0],
                [jnp.sin(angle), jnp.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        box = OrientedBoxGroup(
            centers=[[0, 0, 0]],
            half_extents=[[1, 1, 1]],
            rotations=[rotation],
        )

        ray_origin = jnp.array([5.0, 0.0, 0.0])
        ray_direction = jnp.array([-1.0, 0.0, 0.0])

        t = box.intersect(ray_origin, ray_direction)

        # Rotated 45 degrees, box corner extends sqrt(2) along x
        expected_t = 5.0 - jnp.sqrt(2)
        assert jnp.isclose(t, expected_t, atol=1e-6)


class TestMultipleObstructions:
    """Test behavior with multiple obstructions in a group."""

    def test_returns_nearest_intersection(self):
        """intersect returns the nearest hit across all primitives."""
        cylinders = CylinderGroup(
            p1=[[0, 0, 0], [0, 0, 0]],
            p2=[[0, 0, 10], [0, 0, 10]],
            r=[0.5, 2.0],  # One small, one large
        )

        # Ray from side - should hit larger cylinder first
        ray_origin = jnp.array([5.0, 0.0, 5.0])
        ray_direction = jnp.array([-1.0, 0.0, 0.0])

        t = cylinders.intersect(ray_origin, ray_direction)

        # Hits larger cylinder (r=2) first at t=3
        assert jnp.isclose(t, 3.0, atol=1e-6)
