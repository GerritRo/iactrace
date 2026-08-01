import jax
import numpy as np
import trimesh

from ..core import euler_to_matrix


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


def _color_path(path, rgba):
    """Colour every entity of a ``Path3D`` so the colour survives to the viewer.

    ``Path.colors`` is stored *per entity* and reads back ``None`` until it is
    assigned, in which case the glTF exporter emits no ``COLOR_0`` attribute and
    three.js falls back to the default material -- which is why an uncoloured
    path renders white in a notebook however it was constructed.
    """
    path.colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(path.entities), 1))
    return path


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


def _spin_z(angle):
    """3x3 rotation of ``angle`` radians about +Z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def _replicate_mesh(mesh, rotation, offsets):
    """Copy ``mesh`` to every offset under one shared rotation, as a single mesh.

    Instancing by hand: rotate the template's vertices once, broadcast them over
    the ``(P, 3)`` ``offsets``, and shift the face indices per copy. A camera has
    thousands of pixels, so merging with :func:`trimesh.util.concatenate` per
    pixel is far too slow -- and ``process=False`` keeps trimesh from trying to
    weld the (deliberately) duplicated vertices.
    """
    verts = np.asarray(mesh.vertices) @ np.asarray(rotation).T  # (V, 3)
    faces = np.asarray(mesh.faces)  # (F, 3)
    n_copies, n_verts = len(offsets), len(verts)

    all_verts = verts[None, :, :] + np.asarray(offsets)[:, None, :]  # (P, V, 3)
    all_faces = faces[None, :, :] + (np.arange(n_copies) * n_verts)[:, None, None]
    return trimesh.Trimesh(
        vertices=all_verts.reshape(-1, 3),
        faces=all_faces.reshape(-1, 3),
        process=False,
    )


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


def _cylinder_axis_frame(p1, p2):
    """Shared axis-frame setup for the cylinder mesh builders.

    Returns ``(height, rotation, center)`` -- rotation aligning local Z onto
    the ``p1 -> p2`` axis, and the midpoint to translate to -- or ``None``
    for a degenerate (near-zero-height) axis.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    direction = p2 - p1
    height = np.linalg.norm(direction)

    if height < 1e-10:
        return None

    rotation = _align_z_to(direction / height)
    center = (p1 + p2) / 2
    return height, rotation, center


def _create_cylinder_mesh(p1, p2, radius, sections=16):
    """Create cylinder mesh between two points."""
    frame = _cylinder_axis_frame(p1, p2)
    if frame is None:
        return None
    height, rotation, center = frame

    # Create cylinder along Z, then transform
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cylinder.apply_transform(_rigid_transform(rotation, center))
    return cylinder


def _create_open_cylinder_mesh(p1, p2, radius, sections=16):
    """Create open cylinder mesh (no end caps) between two points."""
    frame = _cylinder_axis_frame(p1, p2)
    if frame is None:
        return None
    height, rotation, center = frame

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
    mesh.apply_transform(_rigid_transform(rotation, center))
    return mesh


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


def _create_sphere_mesh(center, radius, subdivisions=2):
    """Create sphere mesh at given center with given radius."""
    center = np.asarray(center)

    if radius < 1e-10:
        return None

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


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
