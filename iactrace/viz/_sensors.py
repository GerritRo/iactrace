import numpy as np
import trimesh

from ..core import euler_to_matrix
from ._meshes import (
    _create_disk_mesh,
    _create_lofted_mesh,
    _create_polygon_mesh,
    _make_double_sided,
)
from ._utils import convex_hull_2d as _convex_hull_2d


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


def _pixel_centers_2d(sensor_group):
    """Pixel centres ``(n_pixels, 2)`` in the sensor-local frame.

    The counterpart of :func:`_sensor_grid_segments_2d`: where each pixel's
    detection chain is mounted. Conventions match the filled pixel polygons in
    :func:`iactrace.viz.show_image`.
    """
    from ..camera import HexagonalSensorGroup, SquareSensorGroup

    if isinstance(sensor_group, SquareSensorGroup):
        w, h = sensor_group.width, sensor_group.height
        xs = sensor_group.x0 + (np.arange(w) + 0.5) * sensor_group.dx
        ys = sensor_group.y0 + (np.arange(h) + 0.5) * sensor_group.dy
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.column_stack([xx.ravel(), yy.ravel()])

    if isinstance(sensor_group, HexagonalSensorGroup):
        return np.asarray(sensor_group.hex_centers)

    raise TypeError(f"Unsupported sensor group type: {type(sensor_group).__name__}")


def _sensor_grid_segments_2d(sensor_group):
    """Pixel-boundary line segments for one sensor, in the sensor-local frame.

    Returns an ``(S, 2, 2)`` array of ``S`` segments (each a start/end 2D
    point) tracing the pixel grid, or ``None`` for an unsupported group.
    Square groups yield the full set of grid lines; hexagonal groups yield
    each pixel's six-edge outline (interior edges are drawn twice, which is
    fine for a wireframe overlay). The conventions match the filled pixel
    polygons in :func:`iactrace.viz.show_image`.
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


def _thin_slices(z, rings, detail):
    """Drop meridian slices from a lofted cross-section, keeping the ends."""
    z, rings = np.asarray(z), np.asarray(rings)
    n_keep = max(2, int(round(len(z) * detail)))
    if n_keep >= len(z):
        return z, rings
    idx = np.unique(np.linspace(0, len(z) - 1, n_keep).round().astype(int))
    return z[idx], rings[idx]


def _chain_part_meshes(sensor, *, include_entrance=True, detail=1.0, **kwargs):
    """One pixel's detection-chain geometry, in the pixel-local frame.

    Returns a list of ``(mesh, rgba)`` for the entrance aperture, concentrator
    walls, photocathode surface and photodetector body -- whichever the chain
    actually has. Built once and shared by :func:`show_sensor_chain` (a single
    pixel) and :func:`show_camera` (replicated across every pixel), so both show
    exactly the same geometry.

    ``detail`` scales the tessellation (1.0 = full). One pixel is cheap at full
    detail; a whole camera multiplies it by a few thousand, so
    :func:`show_camera` turns it down.
    """
    entrance_color = kwargs.get("entrance_color", [255, 0, 0, 128])
    cone_color = kwargs.get("cone_color", [135, 206, 235, 255])
    detector_color = kwargs.get("detector_color", [80, 200, 80, 200])
    sensor_color = kwargs.get("sensor_color", [80, 80, 80, 255])

    chain = sensor.chain
    parts = []

    # Entrance aperture polygon at z = 0.
    entrance = _pixel_outline_2d(sensor)
    if include_entrance:
        ent_mesh = _make_double_sided(
            _create_polygon_mesh([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], entrance, sag_fn=None)
        )
        if ent_mesh is not None:
            parts.append((ent_mesh, entrance_color))

    # Concentrator walls (optional; skipped if absent or not drawable).
    if chain.concentrator is not None:
        cross = chain.concentrator.cross_sections()
        if cross is not None:
            cone = _create_lofted_mesh(*_thin_slices(cross[0], cross[1], detail))
            if cone is not None:
                parts.append((cone, cone_color))

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
                sag_fn=surface.sag_fn(),
                resolution=max(8, int(round(32 * detail))),
                radial_resolution=max(2, int(round(8 * detail))),
            )
        )
    if det_mesh is not None:
        parts.append((det_mesh, detector_color))

    # Optional 3D photodetector body (e.g. a PMT tube behind the photocathode),
    # lofted below the detector plane; mirrors the concentrator cross_sections path.
    envelope = chain.photodetector.envelope()
    if envelope is not None:
        z_env, rings_env = envelope
        body = _create_lofted_mesh(
            *_thin_slices(np.asarray(z_env) + chain.detector_z, np.asarray(rings_env), detail)
        )
        if body is not None:
            parts.append((body, sensor_color))

    return parts
