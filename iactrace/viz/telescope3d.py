"""3D telescope visualization using trimesh."""

import numpy as np
import trimesh
import jax

from ..core import euler_to_matrix, AsphericSurface


def show_telescope(telescope, **kwargs):
    """
    Visualize telescope in 3D.
    
    In Jupyter notebooks, displays interactive 3D view via three.js (client-side).
    No server-side OpenGL required.
    
    Args:
        telescope: Telescope object
        **kwargs: Additional options:
            - mirror_color: RGBA color for mirrors (default: light blue)
            - obstruction_color: RGBA color for obstructions (default: gray)
            - sensor_color: RGBA color for sensors (default: red)
    
    Returns:
        trimesh.Scene
    """
    mirror_color = kwargs.get('mirror_color', [135, 206, 235, 200])
    obstruction_color = kwargs.get('obstruction_color', [128, 128, 128, 255])
    sensor_color = kwargs.get('sensor_color', [255, 0, 0, 128])
    
    scene = trimesh.Scene()
    
    # Add mirror groups
    for group in telescope.mirror_groups:
        meshes = _get_mirror_meshes(group)
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.visual.face_colors = mirror_color
            scene.add_geometry(combined)
    
    # Add obstruction groups
    if telescope.obstruction_groups:
        for group in telescope.obstruction_groups:
            meshes = _get_obstruction_meshes(group)
            if meshes:
                combined = trimesh.util.concatenate(meshes)
                combined.visual.face_colors = obstruction_color
                scene.add_geometry(combined)
    
    # Add sensors
    for sensor in telescope.sensors:
        mesh = _get_sensor_mesh(sensor)
        if mesh is not None:
            mesh.visual.face_colors = sensor_color
            scene.add_geometry(mesh)
    
    return scene


def export_mesh(telescope, filename):
    """
    Export telescope geometry to 3D file.
    
    Args:
        telescope: Telescope object
        filename: Output path (.glb, .gltf, .stl, .ply, .obj)
    """
    scene = show_telescope(telescope)
    scene.export(filename)


def _get_mirror_meshes(group):
    """Get list of mirror meshes from group."""
    from ..telescope.mirrors import AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup
    
    positions = np.asarray(group.positions)
    rotations = np.asarray(group.rotations)
    surface = AsphericSurface(group.curvature, group.conic, group.aspheric)
    
    meshes = []
    if isinstance(group, AsphericDiskMirrorGroup):
        radii = np.asarray(group.radii)
        for i in range(len(group)):
            mesh = _create_disk_mesh(positions[i], rotations[i], radii[i], surface)
            if mesh is not None:
                meshes.append(mesh)
    
    elif isinstance(group, AsphericPolygonMirrorGroup):
        vertices = np.asarray(group.vertices)
        for i in range(len(group)):
            mesh = _create_polygon_mesh(positions[i], rotations[i], vertices[i], surface)
            if mesh is not None:
                meshes.append(mesh)
    
    return meshes


def _get_obstruction_meshes(group):
    """Get list of obstruction meshes from group."""
    from ..core.obstructions import CylinderGroup, BoxGroup
    
    meshes = []
    if isinstance(group, CylinderGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        r = np.asarray(group.r)
        for i in range(len(group)):
            mesh = _create_cylinder_mesh(p1[i], p2[i], r[i])
            if mesh is not None:
                meshes.append(mesh)
    
    elif isinstance(group, BoxGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        for i in range(len(group)):
            mesh = _create_box_mesh(p1[i], p2[i])
            if mesh is not None:
                meshes.append(mesh)
    
    return meshes


def _get_sensor_mesh(sensor):
    """Get sensor mesh."""
    from ..sensors import SquareSensor, HexagonalSensor
    
    position = np.asarray(sensor.position)
    rotation = np.asarray(sensor.rotation)
    
    if isinstance(sensor, SquareSensor):
        x0, y0 = sensor.x0, sensor.y0
        x1 = x0 + sensor.width * sensor.dx
        y1 = y0 + sensor.height * sensor.dy
        vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return _create_polygon_mesh(position, rotation, vertices, surface=None)
    
    elif isinstance(sensor, HexagonalSensor):
        centers = np.asarray(sensor.hex_centers)
        boundary = _convex_hull_2d(centers)
        if boundary is not None:
            return _create_polygon_mesh(position, rotation, boundary, surface=None)
    
    return None


def _convex_hull_2d(points):
    """Compute convex hull of 2D points."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices]
    except (ImportError, Exception):
        return None


def _create_disk_mesh(position, rotation_euler, radius, surface,
                      resolution=32, radial_resolution=8):
    """Create disk mesh with surface curvature."""
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    radii_vals = np.linspace(0, radius, radial_resolution + 1)
    
    # Build vertices vectorized: center + rings
    # Total vertices = 1 (center) + radial_resolution * resolution
    n_ring_verts = radial_resolution * resolution
    
    # Create ring vertex coordinates
    r_grid, t_grid = np.meshgrid(radii_vals[1:], theta, indexing='ij')
    x_ring = (r_grid * np.cos(t_grid)).ravel()
    y_ring = (r_grid * np.sin(t_grid)).ravel()
    
    # Compute z for all points at once using vmap
    x_all = np.concatenate([[0.0], x_ring])
    y_all = np.concatenate([[0.0], y_ring])
    z_all = np.asarray(jax.vmap(surface.sag)(x_all, y_all))
    
    vertices = np.column_stack([x_all, y_all, z_all])
    
    # Build faces (triangles only)
    faces = []
    
    # Center fan: connect center to first ring
    for i in range(resolution):
        v1 = i + 1
        v2 = (i + 1) % resolution + 1
        faces.append([0, v1, v2])
    
    # Ring-to-ring: connect adjacent rings
    for ring in range(radial_resolution - 1):
        ring_start = 1 + ring * resolution
        next_ring_start = ring_start + resolution
        
        for i in range(resolution):
            v0 = ring_start + i
            v1 = ring_start + (i + 1) % resolution
            v2 = next_ring_start + i
            v3 = next_ring_start + (i + 1) % resolution
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    faces = np.array(faces)
    
    # Transform to world coordinates
    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    world_vertices = vertices @ rot_matrix.T + position
    
    return trimesh.Trimesh(vertices=world_vertices, faces=faces)


def _create_polygon_mesh(position, rotation_euler, vertices_2d, surface,
                         grid_resolution=8):
    """Create polygon mesh with optional surface curvature."""
    vertices_2d = np.asarray(vertices_2d)
    n_verts = len(vertices_2d)
    
    if surface is None:
        # Flat polygon: use fan triangulation
        local_verts = np.zeros((n_verts, 3))
        local_verts[:, :2] = vertices_2d
        
        # Fan triangulation from vertex 0
        faces = np.array([[0, i, i + 1] for i in range(1, n_verts - 1)])
    else:
        # Curved surface: grid + Delaunay triangulation
        xmin, ymin = vertices_2d.min(axis=0)
        xmax, ymax = vertices_2d.max(axis=0)
        
        x_grid = np.linspace(xmin, xmax, grid_resolution)
        y_grid = np.linspace(ymin, ymax, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Filter to points inside polygon
        inside = _points_in_polygon(grid_points, vertices_2d)
        interior_points = grid_points[inside]
        
        # Combine boundary and interior
        all_points_2d = np.vstack([vertices_2d, interior_points])
        
        # Compute z from surface - vectorized with vmap
        z = np.asarray(jax.vmap(surface.sag)(all_points_2d[:, 0], all_points_2d[:, 1]))
        local_verts = np.column_stack([all_points_2d, z])
        
        # Delaunay triangulation
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(all_points_2d)
            
            # Filter triangles to those inside polygon
            centroids = all_points_2d[tri.simplices].mean(axis=1)
            inside_mask = _points_in_polygon(centroids, vertices_2d)
            faces = tri.simplices[inside_mask]
        except ImportError:
            # Fallback: fan triangulation on boundary only
            local_verts = np.zeros((n_verts, 3))
            local_verts[:, :2] = vertices_2d
            local_verts[:, 2] = np.asarray(jax.vmap(surface.sag)(vertices_2d[:, 0], vertices_2d[:, 1]))
            faces = np.array([[0, i, i + 1] for i in range(1, n_verts - 1)])
    
    if len(faces) == 0:
        return None
    
    # Transform to world coordinates
    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    world_verts = local_verts @ rot_matrix.T + position
    
    return trimesh.Trimesh(vertices=world_verts, faces=faces)


def _points_in_polygon(points, vertices):
    """Vectorized point-in-polygon test using ray casting."""
    points = np.asarray(points)
    vertices = np.asarray(vertices)
    n = len(vertices)
    
    x = points[:, 0]
    y = points[:, 1]
    
    inside = np.zeros(len(points), dtype=bool)
    
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        # Vectorized condition check
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        inside = inside ^ cond
        j = i
    
    return inside


def _create_cylinder_mesh(p1, p2, radius, sections=16):
    """Create cylinder mesh between two points."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    direction = p2 - p1
    height = np.linalg.norm(direction)
    
    if height < 1e-10:
        return None
    
    # Create cylinder along Z, then transform
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    
    # Compute transform: rotate Z to direction, translate to midpoint
    direction_norm = direction / height
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Rotation from Z to direction using Rodrigues formula
    if np.allclose(direction_norm, z_axis):
        rotation = np.eye(3)
    elif np.allclose(direction_norm, -z_axis):
        rotation = np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(z_axis, direction_norm)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction_norm)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rotation = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    
    center = (p1 + p2) / 2
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    
    cylinder.apply_transform(transform)
    return cylinder


def _create_box_mesh(p1, p2):
    """Create box mesh from two corner points."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    
    extents = np.abs(p2 - p1)
    center = (p1 + p2) / 2
    
    if np.any(extents < 1e-10):
        return None
    
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def add_rays(scene, origins, directions, length=10.0, color=[255, 255, 0, 255]):
    """
    Add rays to scene for debugging.
    
    Args:
        scene: trimesh.Scene
        origins: Ray origins (N, 3)
        directions: Ray directions (N, 3)
        length: Ray length
        color: RGBA color (unused, trimesh paths don't support colors well)
    
    Returns:
        scene
    """
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    endpoints = origins + directions * length
    n_rays = len(origins)
    
    # Create line segments: interleave origins and endpoints
    vertices = np.empty((2 * n_rays, 3))
    vertices[0::2] = origins
    vertices[1::2] = endpoints
    
    # Create line entities
    entities = [trimesh.path.entities.Line([2 * i, 2 * i + 1]) for i in range(n_rays)]
    
    path = trimesh.path.Path3D(entities=entities, vertices=vertices)
    scene.add_geometry(path)
    return scene


def add_points(scene, points, color=[0, 255, 0, 255]):
    """
    Add points to scene.
    
    Args:
        scene: trimesh.Scene
        points: Point coordinates (N, 3)
        color: RGBA color
    
    Returns:
        scene
    """
    points = np.asarray(points)
    
    cloud = trimesh.PointCloud(points, colors=np.tile(color, (len(points), 1)))
    scene.add_geometry(cloud)
    return scene