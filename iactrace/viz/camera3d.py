"""3D visualization of Camera objects (sensors and concentrators).

Renders sensor planes with pixel grids and optional Winston cone concentrators
using trimesh.  In Jupyter notebooks the result is an interactive three.js view.
"""

import math

import numpy as np
import trimesh

from ..core import euler_to_matrix
from ._utils import convex_hull_2d as _convex_hull_2d


def show_camera(camera, **kwargs):
    """Visualize a Camera in 3D.

    Renders each sensor with its pixel grid and, when present, the
    concentrator funnels at each pixel entrance.

    Args:
        camera: Camera object.
        **kwargs: Visual options:
            - sensor_color: RGBA for sensor plane (default: red, semi-transparent)
            - pixel_grid_color: RGBA for pixel grid lines (default: dark gray)
            - concentrator_color: RGBA for concentrator walls (default: amber)
            - show_pixels: Whether to draw pixel outlines (default: True)
            - show_concentrator: Whether to draw concentrators (default: True)
            - pixel_line_radius: Radius of pixel grid lines (default: auto)

    Returns:
        trimesh.Scene
    """
    sensor_color = kwargs.get("sensor_color", [255, 50, 50, 100])
    pixel_grid_color = kwargs.get("pixel_grid_color", [60, 60, 60, 200])
    concentrator_color = kwargs.get("concentrator_color", [255, 180, 50, 160])
    show_pixels = kwargs.get("show_pixels", True)
    show_concentrator = kwargs.get("show_concentrator", True)
    pixel_line_radius = kwargs.get("pixel_line_radius", None)

    scene = trimesh.Scene()

    for sensor_group in camera.sensor_groups:
        # Sensor plane(s)
        plane_meshes = _get_sensor_plane_meshes(sensor_group)
        for m in plane_meshes:
            m.visual.face_colors = sensor_color
            scene.add_geometry(m)

        # Pixel grid
        if show_pixels:
            grid_meshes = _get_pixel_grid_meshes(
                sensor_group, line_radius=pixel_line_radius
            )
            for m in grid_meshes:
                m.visual.face_colors = pixel_grid_color
                scene.add_geometry(m)

    # Concentrator
    if show_concentrator and camera.concentrator is not None:
        conc_meshes = _get_concentrator_meshes(
            camera.concentrator, camera.sensor_groups
        )
        for m in conc_meshes:
            m.visual.face_colors = concentrator_color
            scene.add_geometry(m)

    return scene


# Sensor planes (same as telescope3d but duplicated to keep module self-contained)


def _get_sensor_plane_meshes(sensor_group):
    """Create flat sensor plane meshes (one per sensor in the group)."""
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    positions = np.asarray(sensor_group.positions)
    rotations = np.asarray(sensor_group.rotations)

    meshes = []

    if isinstance(sensor_group, SquareSensorGroup):
        x0, y0 = sensor_group.x0, sensor_group.y0
        x1 = x0 + sensor_group.width * sensor_group.dx
        y1 = y0 + sensor_group.height * sensor_group.dy
        vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        for i in range(len(positions)):
            mesh = _flat_polygon_mesh(positions[i], rotations[i], vertices)
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(sensor_group, HexagonalSensorGroup):
        centers = np.asarray(sensor_group.hex_centers)
        boundary = _convex_hull_2d(centers)
        if boundary is not None:
            for i in range(len(positions)):
                mesh = _flat_polygon_mesh(positions[i], rotations[i], boundary)
                if mesh is not None:
                    meshes.append(mesh)

    return meshes


# Pixel grid lines


def _get_pixel_grid_meshes(sensor_group, line_radius=None):
    """Create thin cylinder meshes for pixel grid lines."""
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    positions = np.asarray(sensor_group.positions)
    rotations = np.asarray(sensor_group.rotations)

    meshes = []

    if isinstance(sensor_group, SquareSensorGroup):
        for i in range(len(positions)):
            m = _square_grid_mesh(
                sensor_group, positions[i], rotations[i], line_radius
            )
            meshes.extend(m)

    elif isinstance(sensor_group, HexagonalSensorGroup):
        for i in range(len(positions)):
            m = _hex_grid_mesh(
                sensor_group, positions[i], rotations[i], line_radius
            )
            meshes.extend(m)

    return meshes


def _square_grid_mesh(sensor, position, rotation_euler, line_radius=None):
    """Build grid-line cylinders for a square sensor at *position*."""
    x0, y0 = sensor.x0, sensor.y0
    x1 = x0 + sensor.width * sensor.dx
    y1 = y0 + sensor.height * sensor.dy

    if line_radius is None:
        line_radius = min(sensor.dx, sensor.dy) * 0.02

    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    segments = []

    # Vertical lines (along y)
    for col in range(sensor.width + 1):
        x = x0 + col * sensor.dx
        p1_local = np.array([x, y0, 0.0])
        p2_local = np.array([x, y1, 0.0])
        p1 = rot_matrix @ p1_local + position
        p2 = rot_matrix @ p2_local + position
        segments.append((p1, p2))

    # Horizontal lines (along x)
    for row in range(sensor.height + 1):
        y = y0 + row * sensor.dy
        p1_local = np.array([x0, y, 0.0])
        p2_local = np.array([x1, y, 0.0])
        p1 = rot_matrix @ p1_local + position
        p2 = rot_matrix @ p2_local + position
        segments.append((p1, p2))

    return _segments_to_cylinders(segments, line_radius)


def _hex_grid_mesh(sensor, position, rotation_euler, line_radius=None):
    """Build hexagonal cell outlines for a hex sensor at *position*."""
    centers = np.asarray(sensor.hex_centers)
    size = float(sensor.hex_size)

    if line_radius is None:
        line_radius = size * 0.02

    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))

    # Pointy-top hexagon vertex offsets
    angles = np.array([math.pi / 6 + i * math.pi / 3 for i in range(6)])
    vx = size * np.cos(angles)
    vy = size * np.sin(angles)

    # Collect unique edges using a set of sorted endpoint tuples
    edge_set = set()
    segments = []

    for cx, cy in centers:
        hex_verts = np.column_stack([cx + vx, cy + vy])
        for j in range(6):
            k = (j + 1) % 6
            a = (round(hex_verts[j, 0], 8), round(hex_verts[j, 1], 8))
            b = (round(hex_verts[k, 0], 8), round(hex_verts[k, 1], 8))
            edge_key = (min(a, b), max(a, b))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                p1_local = np.array([hex_verts[j, 0], hex_verts[j, 1], 0.0])
                p2_local = np.array([hex_verts[k, 0], hex_verts[k, 1], 0.0])
                p1 = rot_matrix @ p1_local + position
                p2 = rot_matrix @ p2_local + position
                segments.append((p1, p2))

    return _segments_to_cylinders(segments, line_radius)


# Concentrator visualization


def _get_concentrator_meshes(concentrator, sensor_groups):
    """Create meshes for concentrator funnels.

    Supports HexagonalCPC.  A hexagonal funnel (matching the CPC's
    cylindro-parabolic wall geometry) is drawn at every pixel centre
    for all sensor groups.  The sensor shape determines the packing —
    where concentrators sit — while each cone follows the CPC profile.
    """
    from ..camera.concentrator import HexagonalCPC

    if not isinstance(concentrator, HexagonalCPC):
        return []

    exit_inr = float(concentrator.exit_inradius)
    height = float(concentrator.height)

    # Sample the CPC wall profile at several z-slices
    n_slices = 12
    zs = np.linspace(0, height, n_slices)
    C, S, P, Q, T = (float(x) for x in concentrator.cpc_consts)
    radii = np.array([float(_cpc_wall_distance_np(
        z, exit_inr, C, S, P, Q, T,
    )) for z in zs])

    meshes = []

    for sg in sensor_groups:
        centers = _get_pixel_centers(sg)
        if centers is None:
            continue

        positions = np.asarray(sg.positions)
        rotations = np.asarray(sg.rotations)

        for si in range(len(positions)):
            rot_matrix = np.asarray(euler_to_matrix(rotations[si]))
            sensor_pos = positions[si]

            for cx, cy in centers:
                mesh = _create_hex_funnel_mesh(
                    cx, cy, zs, radii, rot_matrix, sensor_pos
                )
                if mesh is not None:
                    meshes.append(mesh)

    return meshes


def _get_pixel_centers(sensor_group):
    """Return (N, 2) array of pixel centres for any sensor group type."""
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    if isinstance(sensor_group, HexagonalSensorGroup):
        return np.asarray(sensor_group.hex_centers)

    if isinstance(sensor_group, SquareSensorGroup):
        cols = np.arange(sensor_group.width) + 0.5
        rows = np.arange(sensor_group.height) + 0.5
        cx = sensor_group.x0 + cols * sensor_group.dx
        cy = sensor_group.y0 + rows * sensor_group.dy
        gx, gy = np.meshgrid(cx, cy)
        return np.column_stack([gx.ravel(), gy.ravel()])

    return None


def _cpc_wall_distance_np(z, a, C, S, P, Q, T):
    """Pure-numpy CPC wall distance (mirrors the JAX version)."""
    A_q = C * C
    B_q = 2.0 * (C * S * z + a * P * P)
    C_q = S * S * z * z - 2.0 * a * z * C * Q - a * a * P * T
    disc = B_q * B_q - 4.0 * A_q * C_q
    return (-B_q + math.sqrt(max(disc, 0.0))) / (2.0 * A_q)


def _create_hex_funnel_mesh(cx, cy, zs, radii, rot_matrix, sensor_pos,
                             n_sides=6):
    """Create a hexagonal funnel mesh at pixel centre (cx, cy).

    The funnel is a stack of hexagonal rings whose inradius varies with z
    according to the CPC profile.  z=0 is the exit (sensor) plane, z=height
    is the entrance.  This matches the real hexagonal CPC geometry used in
    IACT cameras, where each wall is a cylindro-parabolic surface.
    """
    n_slices = len(zs)
    angles = np.array([math.pi / 6 + i * 2 * math.pi / n_sides for i in range(n_sides)])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Hexagon inradius → circumradius conversion: R = inradius / cos(30°)
    circum_factor = 1.0 / math.cos(math.pi / 6)

    all_verts = []
    for k, (z, inr) in enumerate(zip(zs, radii)):
        R = inr * circum_factor
        for j in range(n_sides):
            local = np.array([cx + R * cos_a[j], cy + R * sin_a[j], z])
            world = rot_matrix @ local + sensor_pos
            all_verts.append(world)

    vertices = np.array(all_verts)

    # Build faces between adjacent rings (quads → two triangles)
    faces = []
    for k in range(n_slices - 1):
        base = k * n_sides
        next_base = (k + 1) * n_sides
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            v0 = base + j
            v1 = base + j_next
            v2 = next_base + j
            v3 = next_base + j_next
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    if len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    # Make double-sided
    mesh.faces = np.vstack([mesh.faces, mesh.faces[:, ::-1]])
    return mesh


# Shared helpers


def _flat_polygon_mesh(position, rotation_euler, vertices_2d):
    """Create a flat polygon mesh at *position* with given Euler rotation."""
    vertices_2d = np.asarray(vertices_2d)
    n = len(vertices_2d)
    local = np.zeros((n, 3))
    local[:, :2] = vertices_2d

    faces = np.array([[0, i, i + 1] for i in range(1, n - 1)])

    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    world = local @ rot_matrix.T + position

    mesh = trimesh.Trimesh(vertices=world, faces=faces)
    # double-sided
    mesh.faces = np.vstack([mesh.faces, mesh.faces[:, ::-1]])
    return mesh


def _segments_to_cylinders(segments, radius, sections=6):
    """Convert a list of (p1, p2) pairs to thin cylinder meshes."""
    meshes = []
    for p1, p2 in segments:
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-12:
            continue
        cyl = trimesh.creation.cylinder(
            radius=radius, height=length, sections=sections
        )
        # Align Z axis of cylinder to segment direction
        d_norm = d / length
        z_axis = np.array([0.0, 0.0, 1.0])

        if np.allclose(d_norm, z_axis):
            rot = np.eye(3)
        elif np.allclose(d_norm, -z_axis):
            rot = np.diag([1.0, -1.0, -1.0])
        else:
            v = np.cross(z_axis, d_norm)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, d_norm)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ])
            rot = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[:3, 3] = (p1 + p2) / 2
        cyl.apply_transform(tf)
        meshes.append(cyl)
    return meshes
