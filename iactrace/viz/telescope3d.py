import jax
import numpy as np
import trimesh

from ..core import euler_to_matrix
from ._utils import convex_hull_2d as _convex_hull_2d


def _apply_color(mesh, rgba):
    """Apply RGBA color to a mesh with correct transparency for three.js/glTF rendering.

    glTF materials default to alphaMode='OPAQUE', which causes three.js to ignore
    the alpha channel even when face colors have alpha < 255. Using PBRMaterial with
    alphaMode='BLEND' fixes this for semi-transparent colors.
    """
    rgba = list(rgba)
    if rgba[3] < 255:
        factor = [c / 255.0 for c in rgba]
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=factor,
            alphaMode="BLEND",
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    else:
        mesh.visual.face_colors = rgba


def show_telescope(telescope, camera=None, **kwargs):
    """
    Visualize telescope in 3D.

    In Jupyter notebooks, displays interactive 3D view via three.js (client-side).
    No server-side OpenGL required.

    When a ``camera`` is supplied, each sensor is drawn as a filled face plus a
    black wireframe of its pixel grid (square cells or hexagons), so the
    physical pixel layout can be checked against the rest of the optics.

    Args:
        telescope: Telescope object
        camera: Optional Camera object (for rendering sensors)
        **kwargs: Additional options:
            - mirror_color: RGBA color for mirrors (default: light blue)
            - obstruction_color: RGBA color for obstructions (default: gray)
            - sensor_color: RGBA color for sensors (default: red)
            - lens_color: RGBA color for lenses (default: light green, semi-transparent)
            - show_sensor_grid: Draw per-pixel grid lines for each sensor
              (default: True; only has an effect when ``camera`` is given)
            - sensor_grid_color: RGBA color for the pixel grid (default: black)

    Returns:
        trimesh.Scene
    """
    mirror_color = kwargs.get("mirror_color", [135, 206, 235, 200])
    obstruction_color = kwargs.get("obstruction_color", [128, 128, 128, 255])
    sensor_color = kwargs.get("sensor_color", [255, 0, 0, 128])
    lens_color = kwargs.get("lens_color", [144, 238, 144, 150])  # Light green, semi-transparent
    show_sensor_grid = kwargs.get("show_sensor_grid", True)
    sensor_grid_color = kwargs.get("sensor_grid_color", [0, 0, 0, 255])  # Black pixel grid

    scene = trimesh.Scene()

    # Each optical family renders the same way: build per-element meshes,
    # merge them, colour the merge, and add it to the scene.
    for groups, get_meshes, color in (
        (telescope.mirror_groups, _get_mirror_meshes, mirror_color),
        (telescope.lens_groups, _get_lens_meshes, lens_color),
        (telescope.obstruction_groups, _get_obstruction_meshes, obstruction_color),
    ):
        for group in groups:
            meshes = get_meshes(group)
            if meshes:
                combined = trimesh.util.concatenate(meshes)
                _apply_color(combined, color)
                scene.add_geometry(combined)

    # Add sensors from camera
    if camera is not None:
        cam_transform = _rigid_transform(
            np.asarray(euler_to_matrix(telescope.camera_rotation)),
            np.asarray(telescope.camera_position),
        )
        for sensor_group in camera.sensor_groups:
            meshes = _get_sensor_meshes(sensor_group)
            for mesh in meshes:
                mesh.apply_transform(cam_transform)
                _apply_color(mesh, sensor_color)
                scene.add_geometry(mesh)

            if show_sensor_grid:
                for path in _get_sensor_grid_paths(sensor_group, sensor_grid_color):
                    path.apply_transform(cam_transform)
                    scene.add_geometry(path)

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


def show_sensor_chain(camera, sensor_idx=0, **kwargs):
    """Visualize a single pixel's detection chain ("train") in 3D.

    Draws, in the canonical pixel-local frame (entrance aperture at ``z = 0``,
    axis ``+z``): the pixel entrance aperture, the concentrator walls (if a
    concentrator is present and exposes :meth:`Concentrator.cross_sections`), and
    the photodetector's sensor surface -- its actual photocathode geometry, drawn
    curved when the photodetector owns a curved
    :class:`~iactrace.camera.detector.surface.DetectionSurface`, otherwise a
    flat active-area polygon at ``chain.detector_z``.

    If the photodetector exposes a 3D envelope (:meth:`PhotoDetector.envelope`, e.g.
    a :class:`~iactrace.camera.detector.pmt.PMT`), its glass body is lofted around
    the detector plane as well.

    Args:
        camera: Camera object (the selected sensor group and its ``chain`` are read).
        sensor_idx: Which sensor group to take the pixel geometry and chain from.
        **kwargs: ``entrance_color`` / ``cone_color`` / ``detector_color`` /
            ``sensor_color`` RGBA.

    Returns:
        trimesh.Scene
    """
    entrance_color = kwargs.get("entrance_color", [255, 0, 0, 128])
    cone_color = kwargs.get("cone_color", [135, 206, 235, 160])
    detector_color = kwargs.get("detector_color", [80, 80, 80, 255])
    sensor_color = kwargs.get("sensor_color", [200, 200, 255, 90])

    sensor = camera.sensor_groups[sensor_idx]
    chain = sensor.chain
    scene = trimesh.Scene()

    # Entrance aperture polygon at z = 0.
    entrance = _pixel_outline_2d(sensor)
    ent_mesh = _make_double_sided(
        _create_polygon_mesh([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], entrance, sag_fn=None)
    )
    if ent_mesh is not None:
        _apply_color(ent_mesh, entrance_color)
        scene.add_geometry(ent_mesh)

    # Concentrator walls (optional; skipped if absent or not drawable).
    if chain.concentrator is not None:
        cross = chain.concentrator.cross_sections()
        if cross is not None:
            z, rings = cross
            cone = _create_lofted_mesh(z, rings)
            if cone is not None:
                _apply_color(cone, cone_color)
                scene.add_geometry(cone)

    # Photocathode stopping surface, at its own vertex position (light enters at
    # z=0 from +z and the chain runs toward -z; the surface sits near the detector
    # plane and may bulge up into a cone). The photodetector owns this geometry.
    surface = chain.surface
    if surface.is_flat:
        # Flat detector: draw the active-area polygon (outline / fallback ring).
        det_outline = chain.photodetector.outline()
        if det_outline is None:
            cross = chain.concentrator.cross_sections() if chain.concentrator is not None else None
            if cross is not None:
                exit_ring = np.asarray(cross[1][-1])  # (M, 2) exit aperture
                r_det = float(np.max(np.linalg.norm(exit_ring, axis=-1)))
                ang = np.linspace(0.0, 2.0 * np.pi, 41)[:-1]
                det_outline = np.column_stack([r_det * np.cos(ang), r_det * np.sin(ang)])
            else:
                det_outline = entrance
        det_mesh = _make_double_sided(
            _create_polygon_mesh(
                [0.0, 0.0, surface.vertex_z], [0.0, 0.0, 0.0], np.asarray(det_outline), sag_fn=None
            )
        )
    else:
        # Curved photocathode: draw the actual surface of revolution from its sag.
        r_det = surface.radius
        if not np.isfinite(r_det):
            cross = chain.concentrator.cross_sections() if chain.concentrator is not None else None
            r_det = (
                float(np.max(np.linalg.norm(np.asarray(cross[1][-1]), axis=-1)))
                if cross is not None
                else float(np.max(np.linalg.norm(np.asarray(entrance), axis=-1)))
            )
        det_mesh = _make_double_sided(
            _create_disk_mesh(
                [0.0, 0.0, surface.vertex_z],
                [0.0, 0.0, 0.0],
                float(r_det),
                sag_fn=surface.shape._index(0)._sag_local,
            )
        )
    if det_mesh is not None:
        _apply_color(det_mesh, detector_color)
        scene.add_geometry(det_mesh)

    # Optional 3D photodetector body (e.g. a PMT tube behind the photocathode),
    # lofted below the detector plane; mirrors the concentrator cross_sections path.
    envelope = chain.photodetector.envelope()
    if envelope is not None:
        z_env, rings_env = envelope
        body = _create_lofted_mesh(np.asarray(z_env) + chain.detector_z, np.asarray(rings_env))
        if body is not None:
            _apply_color(body, sensor_color)
            scene.add_geometry(body)

    return scene


def _make_double_sided(mesh):
    """Make mesh visible from both sides by duplicating faces with reversed winding."""
    if mesh is None:
        return None
    faces_reversed = mesh.faces[:, ::-1]
    mesh.faces = np.vstack([mesh.faces, faces_reversed])
    return mesh

def _rigid_transform(rotation, translation):
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and a translation."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def _align_z_to(direction_norm):
    """Rotation mapping local +Z onto the unit ``direction_norm`` (Rodrigues)."""
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(direction_norm, z_axis):
        return np.eye(3)
    if np.allclose(direction_norm, -z_axis):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z_axis, direction_norm)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction_norm)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

def _curved_face_meshes(group):
    """One double-sided curved face mesh per element.

    Shared by mirror groups and refractive lens groups: each element renders
    as a single surface honouring its own aperture and sag.
    """
    meshes = []
    for i in range(len(group)):

        def sag_fn(x, y, _i=i):
            return group.surface.sag_at(_i, x, y)

        mesh = _aperture_face_mesh(
            group.positions[i], group.rotations[i], group.aperture, i, sag_fn=sag_fn
        )
        if mesh is not None:
            meshes.append(_make_double_sided(mesh))
    return meshes


def _get_mirror_meshes(group):
    """Curved face mesh per mirror (each with its own curvature/conic/aspherics)."""
    return _curved_face_meshes(group)


def _get_lens_meshes(group):
    """Get list of lens meshes from group.

    Refractive elements render as a single curved face (one per element);
    slab elements render as a volume extruded along the local Z-axis.
    Both honour the element's aperture, so polygonal lenses and windows
    are supported alongside circular ones.
    """
    from ..core.interactions import RefractInteraction, SlabInteraction

    if isinstance(group.interaction_module, RefractInteraction):
        return _curved_face_meshes(group)

    if isinstance(group.interaction_module, SlabInteraction):
        thickness = np.asarray(group.interaction_module.thickness)
        meshes = []
        for i in range(len(group)):
            mesh = _aperture_slab_mesh(
                group.positions[i], group.rotations[i], group.aperture, i, float(thickness[i])
            )
            if mesh is not None:
                meshes.append(mesh)
        return meshes

    return []


def _get_obstruction_meshes(group):
    """Get list of obstruction meshes from group."""
    from ..core.obstructions import (
        BoxGroup,
        CylinderGroup,
        OpenCylinderGroup,
        OrientedBoxGroup,
        SphereGroup,
        TriangleGroup,
    )

    meshes = []
    if isinstance(group, CylinderGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        r = np.asarray(group.r)
        for i in range(len(group)):
            mesh = _create_cylinder_mesh(p1[i], p2[i], r[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, OpenCylinderGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        r = np.asarray(group.r)
        for i in range(len(group)):
            mesh = _create_open_cylinder_mesh(p1[i], p2[i], r[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, BoxGroup):
        p1 = np.asarray(group.p1)
        p2 = np.asarray(group.p2)
        for i in range(len(group)):
            mesh = _create_box_mesh(p1[i], p2[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, SphereGroup):
        centers = np.asarray(group.centers)
        radii = np.asarray(group.radii)
        for i in range(len(group)):
            mesh = _create_sphere_mesh(centers[i], radii[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, OrientedBoxGroup):
        centers = np.asarray(group.centers)
        half_extents = np.asarray(group.half_extents)
        rotations = np.asarray(group.rotations)
        for i in range(len(group)):
            mesh = _create_oriented_box_mesh(centers[i], half_extents[i], rotations[i])
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(group, TriangleGroup):
        v0 = np.asarray(group.v0)
        v1 = np.asarray(group.v1)
        v2 = np.asarray(group.v2)
        for i in range(len(group)):
            mesh = _create_triangle_mesh(v0[i], v1[i], v2[i])
            if mesh is not None:
                meshes.append(mesh)

    return meshes


def _get_sensor_meshes(sensor_group):
    """Get list of sensor meshes from a sensor group.

    Each sensor in the group gets its own mesh, rendered at its position/rotation.
    """
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    positions = np.asarray(sensor_group.positions)
    rotations = np.asarray(sensor_group.rotations)

    meshes = []
    n_sensors = len(sensor_group)

    if isinstance(sensor_group, SquareSensorGroup):
        # Compute 2D boundary vertices from pixel geometry
        x0, y0 = sensor_group.x0, sensor_group.y0
        x1 = x0 + sensor_group.width * sensor_group.dx
        y1 = y0 + sensor_group.height * sensor_group.dy
        vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        for i in range(n_sensors):
            mesh = _create_polygon_mesh(positions[i], rotations[i], vertices, sag_fn=None)
            mesh = _make_double_sided(mesh)
            if mesh is not None:
                meshes.append(mesh)

    elif isinstance(sensor_group, HexagonalSensorGroup):
        # Use convex hull of hex centers as boundary
        centers = np.asarray(sensor_group.hex_centers)
        boundary = _convex_hull_2d(centers)

        if boundary is not None:
            for i in range(n_sensors):
                mesh = _create_polygon_mesh(positions[i], rotations[i], boundary, sag_fn=None)
                mesh = _make_double_sided(mesh)
                if mesh is not None:
                    meshes.append(mesh)

    return meshes


def _sensor_grid_segments_2d(sensor_group):
    """Pixel-boundary line segments for one sensor, in the sensor-local frame.

    Returns an ``(S, 2, 2)`` array of ``S`` segments (each a start/end 2D
    point) tracing the pixel grid, or ``None`` for an unsupported group.
    Square groups yield the full set of grid lines; hexagonal groups yield
    each pixel's six-edge outline (interior edges are drawn twice, which is
    fine for a wireframe overlay). The conventions match the filled pixel
    polygons in :func:`iactrace.viz.plotting.show_camera`.
    """
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    if isinstance(sensor_group, SquareSensorGroup):
        w, h = sensor_group.width, sensor_group.height
        xs = sensor_group.x0 + np.arange(w + 1) * sensor_group.dx
        ys = sensor_group.y0 + np.arange(h + 1) * sensor_group.dy
        verticals = np.stack(
            [
                np.column_stack([xs, np.full(w + 1, ys[0])]),
                np.column_stack([xs, np.full(w + 1, ys[-1])]),
            ],
            axis=1,
        )  # (w+1, 2, 2)
        horizontals = np.stack(
            [
                np.column_stack([np.full(h + 1, xs[0]), ys]),
                np.column_stack([np.full(h + 1, xs[-1]), ys]),
            ],
            axis=1,
        )  # (h+1, 2, 2)
        return np.concatenate([verticals, horizontals], axis=0)

    if isinstance(sensor_group, HexagonalSensorGroup):
        centers = np.asarray(sensor_group.hex_centers)
        s = sensor_group.hex_size
        angles = np.deg2rad(np.arange(30.0, 360.0, 60.0)) + sensor_group.grid_rotation
        offsets = s * np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (6, 2)
        corners = centers[:, None, :] + offsets[None, :, :]  # (n_pix, 6, 2)
        nxt = np.roll(corners, -1, axis=1)
        segments = np.stack([corners, nxt], axis=2)  # (n_pix, 6, 2, 2)
        return segments.reshape(-1, 2, 2)

    return None


def _get_sensor_grid_paths(sensor_group, color):
    """Build pixel-grid wireframe paths for a sensor group (camera frame).

    Returns one :class:`trimesh.path.Path3D` per sensor, drawing the pixel
    boundaries as coloured line segments so the physical pixel layout can be
    inspected alongside the filled sensor faces. The lines are nudged a hair
    off the sensor plane so they never z-fight with the sensor mesh.
    """
    segments_2d = _sensor_grid_segments_2d(sensor_group)
    if segments_2d is None or len(segments_2d) == 0:
        return []

    # Offset the wireframe by 0.1% of the sensor extent along the local +Z so
    # it sits just in front of the filled face (negligible, avoids z-fighting).
    lift = 1e-3 * float(np.ptp(segments_2d.reshape(-1, 2), axis=0).max())

    n_seg = len(segments_2d)
    local = np.zeros((n_seg, 2, 3))
    local[:, :, :2] = segments_2d
    local[:, :, 2] = lift

    positions = np.asarray(sensor_group.positions)
    rotations = np.asarray(sensor_group.rotations)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (n_seg, 1))

    paths = []
    for i in range(len(sensor_group)):
        rot = np.asarray(euler_to_matrix(rotations[i]))
        world = local @ rot.T + positions[i]
        path = trimesh.path.Path3D(
            entities=[trimesh.path.entities.Line([2 * k, 2 * k + 1]) for k in range(n_seg)],
            vertices=world.reshape(-1, 3),
        )
        path.colors = colors
        paths.append(path)
    return paths


def _pixel_outline_2d(sensor_group):
    """Single-pixel boundary polygon ``(M, 2)``, centred at the origin.

    Grid-aligned (the pixel-local frame), so it lines up with a concentrator's
    ``cross_sections`` and the detector outline drawn by
    :func:`show_sensor_chain`.
    """
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    if isinstance(sensor_group, SquareSensorGroup):
        hx, hy = sensor_group.dx / 2.0, sensor_group.dy / 2.0
        return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    if isinstance(sensor_group, HexagonalSensorGroup):
        r = sensor_group.hex_size
        angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    raise TypeError(f"Unsupported sensor group type: {type(sensor_group).__name__}")


def _create_lofted_mesh(z, rings):
    """Loft a stack of polygon cross-sections into a wall mesh.

    Args:
        z: ``(K,)`` axial heights.
        rings: ``(K, M, 2)`` polygon vertices per slice (pixel-local frame).

    Builds quads between consecutive rings ``rings[k]`` / ``rings[k+1]`` and
    returns a double-sided :class:`trimesh.Trimesh` (walls only, no caps).
    Generalizes :func:`_create_open_cylinder_mesh` to a varying cross-section,
    so it covers hexagonal, square and (large ``M``) round cones alike.
    """
    z = np.asarray(z, dtype=float)
    rings = np.asarray(rings, dtype=float)
    if rings.ndim != 3 or rings.shape[0] < 2 or rings.shape[1] < 3:
        return None
    k_slices, m_verts = rings.shape[0], rings.shape[1]

    vertices = np.zeros((k_slices * m_verts, 3))
    vertices[:, :2] = rings.reshape(k_slices * m_verts, 2)
    vertices[:, 2] = np.repeat(z, m_verts)

    faces = []
    for k in range(k_slices - 1):
        base, nxt = k * m_verts, (k + 1) * m_verts
        for m in range(m_verts):
            m1 = (m + 1) % m_verts
            a, b, c, d = base + m, base + m1, nxt + m, nxt + m1
            faces.append([a, b, d])
            faces.append([a, d, c])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces))
    return _make_double_sided(mesh)


def _create_disk_mesh(
    position,
    rotation_euler,
    radius,
    sag_fn=None,
    inner_radius=0.0,
    resolution=32,
    radial_resolution=8,
):
    """Create disk mesh with surface curvature.

    Args:
        sag_fn: Callable (x, y) -> z for surface height, or None for flat.

    For annulus shapes (inner_radius > 0), creates a ring mesh with a center hole.
    """
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    # Check if we have an annulus (center hole)
    has_inner_radius = inner_radius > 0

    if has_inner_radius:
        # Annulus: radii from inner_radius to outer radius
        radii_vals = np.linspace(inner_radius, radius, radial_resolution + 1)

        # Create ring vertex coordinates (no center vertex)
        r_grid, t_grid = np.meshgrid(radii_vals, theta, indexing="ij")
        x_all = (r_grid * np.cos(t_grid)).ravel()
        y_all = (r_grid * np.sin(t_grid)).ravel()

        z_all = np.asarray(jax.vmap(sag_fn)(x_all, y_all)) if sag_fn else np.zeros_like(x_all)

        vertices = np.column_stack([x_all, y_all, z_all])

        # Build faces (triangles only) - ring-to-ring connections only
        faces = []
        n_rings = radial_resolution + 1

        for ring in range(n_rings - 1):
            ring_start = ring * resolution
            next_ring_start = ring_start + resolution

            for i in range(resolution):
                v0 = ring_start + i
                v1 = ring_start + (i + 1) % resolution
                v2 = next_ring_start + i
                v3 = next_ring_start + (i + 1) % resolution

                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
    else:
        # Full disk: radii from 0 to outer radius
        radii_vals = np.linspace(0, radius, radial_resolution + 1)

        # Create ring vertex coordinates
        r_grid, t_grid = np.meshgrid(radii_vals[1:], theta, indexing="ij")
        x_ring = (r_grid * np.cos(t_grid)).ravel()
        y_ring = (r_grid * np.sin(t_grid)).ravel()

        # Compute z for all points at once using vmap
        x_all = np.concatenate([[0.0], x_ring])
        y_all = np.concatenate([[0.0], y_ring])
        z_all = np.asarray(jax.vmap(sag_fn)(x_all, y_all)) if sag_fn else np.zeros_like(x_all)

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

    # Transform to world coordinates
    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    world_vertices = vertices @ rot_matrix.T + position

    return trimesh.Trimesh(vertices=world_vertices, faces=faces)


def _create_polygon_mesh(position, rotation_euler, vertices_2d, sag_fn=None, grid_resolution=8):
    """Create polygon mesh with optional surface curvature.

    Args:
        sag_fn: Callable (x, y) -> z for surface height, or None for flat.
    """
    vertices_2d = np.asarray(vertices_2d)
    n_verts = len(vertices_2d)

    if sag_fn is None:
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
        z = np.asarray(jax.vmap(sag_fn)(all_points_2d[:, 0], all_points_2d[:, 1]))
        local_verts = np.column_stack([all_points_2d, z])

        # Delaunay triangulation
        try:
            from scipy.spatial import Delaunay  # type: ignore[import-untyped]

            tri = Delaunay(all_points_2d)

            # Filter triangles to those inside polygon
            centroids = all_points_2d[tri.simplices].mean(axis=1)
            inside_mask = _points_in_polygon(centroids, vertices_2d)
            faces = tri.simplices[inside_mask]

        except ImportError:
            # Fallback: fan triangulation on boundary only
            local_verts = np.zeros((n_verts, 3))
            local_verts[:, :2] = vertices_2d
            local_verts[:, 2] = np.asarray(jax.vmap(sag_fn)(vertices_2d[:, 0], vertices_2d[:, 1]))
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

    # Rotate local Z onto the axis and translate to the midpoint.
    rotation = _align_z_to(direction / height)
    center = (p1 + p2) / 2
    cylinder.apply_transform(_rigid_transform(rotation, center))
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


def _create_open_cylinder_mesh(p1, p2, radius, sections=16):
    """Create open cylinder mesh (no end caps) between two points."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    direction = p2 - p1
    height = np.linalg.norm(direction)

    if height < 1e-10:
        return None

    # Create cylinder without caps
    # trimesh.creation.cylinder creates a capped cylinder, so we create an open one manually
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)

    # Bottom and top ring vertices
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    bottom_verts = np.column_stack([x, y, np.full(sections, -height / 2)])
    top_verts = np.column_stack([x, y, np.full(sections, height / 2)])
    vertices = np.vstack([bottom_verts, top_verts])

    # Create faces for the curved surface (quads split into triangles)
    faces = []
    for i in range(sections):
        next_i = (i + 1) % sections
        # Bottom vertex indices: 0 to sections-1
        # Top vertex indices: sections to 2*sections-1
        b0, b1 = i, next_i
        t0, t1 = i + sections, next_i + sections
        # Outward-facing pair.
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
        # Inward-facing pair
        faces.append([b0, t1, b1])
        faces.append([b0, t0, t1])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Rotate local Z onto the axis and translate to the midpoint.
    rotation = _align_z_to(direction / height)
    center = (p1 + p2) / 2
    mesh.apply_transform(_rigid_transform(rotation, center))
    return mesh


def _create_sphere_mesh(center, radius, subdivisions=2):
    """Create sphere mesh at given center with given radius."""
    center = np.asarray(center)

    if radius < 1e-10:
        return None

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def _create_oriented_box_mesh(center, half_extents, rotation_matrix):
    """Create oriented box mesh with given center, half extents, and rotation."""
    center = np.asarray(center)
    half_extents = np.asarray(half_extents)
    rotation_matrix = np.asarray(rotation_matrix)

    extents = 2 * half_extents

    if np.any(extents < 1e-10):
        return None

    box = trimesh.creation.box(extents=extents)
    box.apply_transform(_rigid_transform(rotation_matrix, center))
    return box


def _create_triangle_mesh(v0, v1, v2):
    """Create a single triangle mesh from three vertices."""
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    vertices = np.array([v0, v1, v2])
    faces = np.array([[0, 1, 2]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = _make_double_sided(mesh)
    return mesh


def _aperture_face_mesh(position, rotation_euler, aperture, i, sag_fn=None):
    """Build a single face mesh for element ``i`` of ``aperture``.

    Dispatches on aperture type so callers (mirrors, refractive lenses,
    slab caps) don't need to special-case disks vs. polygons.
    """
    from ..core.apertures import DiskAperture, PolygonAperture

    if isinstance(aperture, DiskAperture):
        return _create_disk_mesh(
            position,
            rotation_euler,
            radius=float(aperture.radii[i]),
            sag_fn=sag_fn,
            inner_radius=float(aperture.inner_radii[i]),
        )
    if isinstance(aperture, PolygonAperture):
        return _create_polygon_mesh(
            position,
            rotation_euler,
            vertices_2d=np.asarray(aperture.vertices[i]),
            sag_fn=sag_fn,
        )
    raise TypeError(f"Unsupported aperture type: {type(aperture).__name__}")


def _aperture_slab_mesh(position, rotation_euler, aperture, i, thickness, sections=32):
    """Extrude element ``i`` of ``aperture`` along the local Z-axis.

    Produces a cylinder for ``DiskAperture`` and a prism for
    ``PolygonAperture``. The front face sits at ``position`` (local
    z = 0) and the back face at local z = +thickness, matching the
    physics convention in :func:`iactrace.core.interactions.refract_slab`
    where ``position`` is the entry point on the front surface.
    """
    from ..core.apertures import DiskAperture, PolygonAperture

    if thickness < 1e-10:
        return None

    if isinstance(aperture, DiskAperture):
        radius = float(aperture.radii[i])
        if radius < 1e-10:
            return None
        mesh = trimesh.creation.cylinder(radius=radius, height=thickness, sections=sections)
        mesh.apply_translation([0.0, 0.0, thickness / 2.0])
    elif isinstance(aperture, PolygonAperture):
        verts_2d = np.asarray(aperture.vertices[i])
        n_verts = len(verts_2d)
        faces_2d = np.array([[0, k, k + 1] for k in range(1, n_verts - 1)])
        mesh = trimesh.creation.extrude_triangulation(verts_2d, faces_2d, height=thickness)
    else:
        raise TypeError(f"Unsupported aperture type: {type(aperture).__name__}")

    rot_matrix = np.asarray(euler_to_matrix(rotation_euler))
    mesh.apply_transform(_rigid_transform(rot_matrix, np.asarray(position)))
    return mesh


def add_rays(scene, origins, directions, length=10.0, color=None):
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
    if color is None:
        color = [255, 255, 0, 255]
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


def add_points(scene, points, color=None):
    """
    Add points to scene.

    Args:
        scene: trimesh.Scene
        points: Point coordinates (N, 3)
        color: RGBA color

    Returns:
        scene
    """
    if color is None:
        color = [0, 255, 0, 255]
    points = np.asarray(points)

    cloud = trimesh.PointCloud(points, colors=np.tile(color, (len(points), 1)))
    scene.add_geometry(cloud)
    return scene
