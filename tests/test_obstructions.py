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

    def test_creation(self):
        """Cylinder group can be created."""
        cylinders = CylinderGroup(
            p1=[[0, 0, 0], [1, 0, 0]],
            p2=[[0, 0, 5], [1, 0, 5]],
            r=[0.1, 0.2],
        )

        assert len(cylinders) == 2

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

    def test_ray_misses_cylinder(self):
        """Ray missing cylinder returns infinity."""
        cylinder = CylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 10]],
            r=[1.0],
        )

        ray_origin = jnp.array([5.0, 0.0, 15.0])  # Above cylinder
        ray_direction = jnp.array([-1.0, 0.0, 0.0])

        t = cylinder.intersect(ray_origin, ray_direction)

        assert jnp.isinf(t)

    def test_to_config(self):
        """Cylinder converts to config dict."""
        cylinder = CylinderGroup(
            p1=[[0, 0, 0]],
            p2=[[0, 0, 5]],
            r=[0.1],
        )

        config = cylinder.to_config(0)

        assert config['type'] == 'cylinder'
        assert config['p1'] == [0.0, 0.0, 0.0]
        assert config['p2'] == [0.0, 0.0, 5.0]
        assert config['r'] == 0.1


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

    def test_creation(self):
        """Box group can be created."""
        boxes = BoxGroup(
            p1=[[0, 0, 0], [5, 5, 5]],
            p2=[[1, 1, 1], [6, 6, 6]],
        )

        assert len(boxes) == 2

    def test_ray_hits_box(self):
        """Ray hitting box returns valid t."""
        box = BoxGroup(
            p1=[[0, 0, 0]],
            p2=[[2, 2, 2]],
        )

        ray_origin = jnp.array([1.0, 1.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = box.intersect(ray_origin, ray_direction)

        # Hits top face at z=2, so t=8
        assert jnp.isclose(t, 8.0, atol=1e-6)

    def test_ray_misses_box(self):
        """Ray missing box returns infinity."""
        box = BoxGroup(
            p1=[[0, 0, 0]],
            p2=[[2, 2, 2]],
        )

        ray_origin = jnp.array([5.0, 5.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = box.intersect(ray_origin, ray_direction)

        assert jnp.isinf(t)

    def test_to_config(self):
        """Box converts to config dict."""
        box = BoxGroup(
            p1=[[0, 0, 0]],
            p2=[[1, 2, 3]],
        )

        config = box.to_config(0)

        assert config['type'] == 'box'
        assert config['p1'] == [0.0, 0.0, 0.0]
        assert config['p2'] == [1.0, 2.0, 3.0]


class TestSphereGroup:
    """Test sphere obstruction group."""

    def test_creation(self):
        """Sphere group can be created."""
        spheres = SphereGroup(
            centers=[[0, 0, 0], [5, 0, 0]],
            radii=[1.0, 2.0],
        )

        assert len(spheres) == 2

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

    def test_ray_misses_sphere(self):
        """Ray missing sphere returns infinity."""
        sphere = SphereGroup(
            centers=[[0, 0, 0]],
            radii=[1.0],
        )

        ray_origin = jnp.array([5.0, 5.0, 10.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = sphere.intersect(ray_origin, ray_direction)

        assert jnp.isinf(t)

    def test_to_config(self):
        """Sphere converts to config dict."""
        sphere = SphereGroup(
            centers=[[1, 2, 3]],
            radii=[0.5],
        )

        config = sphere.to_config(0)

        assert config['type'] == 'sphere'
        assert config['center'] == [1.0, 2.0, 3.0]
        assert config['r'] == 0.5


class TestTriangleGroup:
    """Test triangle obstruction group."""

    def test_creation(self):
        """Triangle group can be created."""
        triangles = TriangleGroup(
            v0=[[0, 0, 0]],
            v1=[[2, 0, 0]],
            v2=[[1, 2, 0]],
        )

        assert len(triangles) == 1

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

    def test_ray_misses_triangle(self):
        """Ray missing triangle returns infinity."""
        triangles = TriangleGroup(
            v0=[[0, 0, 0]],
            v1=[[2, 0, 0]],
            v2=[[1, 2, 0]],
        )

        ray_origin = jnp.array([10.0, 10.0, 5.0])
        ray_direction = jnp.array([0.0, 0.0, -1.0])

        t = triangles.intersect(ray_origin, ray_direction)

        assert jnp.isinf(t)


class TestOrientedBoxGroup:
    """Test oriented box obstruction group."""

    def test_creation(self):
        """Oriented box group can be created."""
        boxes = OrientedBoxGroup(
            centers=[[0, 0, 0]],
            half_extents=[[1, 1, 1]],
            rotations=[jnp.eye(3)],
        )

        assert len(boxes) == 1

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
        rotation = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0]
        ])

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

    def test_from_config_roundtrip(self):
        """Config roundtrip preserves obstruction parameters."""
        original = CylinderGroup(
            p1=[[0, 0, 0], [1, 1, 1]],
            p2=[[0, 0, 5], [1, 1, 6]],
            r=[0.1, 0.2],
        )

        # Convert to config
        configs = [original.to_config(i) for i in range(len(original))]

        # Reconstruct
        reconstructed = CylinderGroup.from_config(configs)

        assert len(reconstructed) == len(original)
        assert jnp.allclose(original.p1, reconstructed.p1)
        assert jnp.allclose(original.p2, reconstructed.p2)
        assert jnp.allclose(original.r, reconstructed.r)
