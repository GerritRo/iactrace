import numpy as np
import trimesh

from ._meshes import _color_path


def add_rays(scene, origins, directions, length=10.0, color=None):
    """
    Add rays to scene for debugging.

    Args:
        scene: trimesh.Scene
        origins: Ray origins (N, 3)
        directions: Ray directions (N, 3)
        length: Ray length
        color: RGBA color for the rays (default: yellow)

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
    _color_path(path, color)
    scene.add_geometry(path)
    return scene


def _clip_segment(p0, p1, lo, hi):
    """Portion of the segment ``p0 -> p1`` inside the box, or ``None``."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-15:  # parallel to this slab
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return None
            continue
        a = (lo[axis] - p0[axis]) / d[axis]
        b = (hi[axis] - p0[axis]) / d[axis]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return p0 + t0 * d, p0 + t1 * d, t0 > 0.0, t1 < 1.0


def _clipped_runs(points, lo, hi):
    """
    Split a polyline into the contiguous runs that lie inside the box.
    """
    if lo is None:
        return [points]
    runs: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    joinable = False
    for p0, p1 in zip(points[:-1], points[1:], strict=True):
        piece = _clip_segment(p0, p1, lo, hi)
        if piece is None:
            if len(current) >= 2:
                runs.append(current)
            current, joinable = [], False
            continue
        q0, q1, cut_start, cut_end = piece
        if joinable and not cut_start:
            current.append(q1)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [q0, q1]
        joinable = not cut_end
    if len(current) >= 2:
        runs.append(current)
    return runs


def add_trajectories(scene, trajectory, color=None, clip=None):
    """Add multi-segment ray paths (polylines) to a scene.

    Draws one polyline per ray through its consecutive positions.

    Args:
        scene: trimesh.Scene
        trajectory: A :class:`~iactrace.core.trajectory.Trajectory` (as returned
            by ``Telescope.trace(..., record_trajectory=True)``), a
            :class:`~iactrace.core.trajectory.TraceResult` or
            :class:`~iactrace.camera.optics.ChainTrace` (or any object exposing a
            ``trajectory`` attribute), or a raw ``(steps + 1, N, 3)`` array.
            ``None`` (recording was off) is a no-op.
        color: RGBA color for the ray paths (default: amber). Applied per
            polyline, so it survives the glTF export a notebook renders through.
        clip: Optional axis-aligned box ``(lo, hi)``, each ``(3,)``, to trim the
            paths to. A ray's leg in from the optics is metres long while a
            camera is centimetres deep, so drawing it whole leaves the subject
            a speck in the corner; clipping keeps the geometry in frame. Paths
            are cut, not filtered -- a ray that only passes through the box
            still contributes the part that is inside.

    Returns:
        scene
    """
    if color is None:
        color = [255, 200, 0, 255]
    # Unwrap a TraceResult / ChainTrace-like container to its trajectory
    trajectory = getattr(trajectory, "trajectory", trajectory)
    if trajectory is None:
        return scene
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 3 or trajectory.shape[0] < 2:
        return scene
    steps, n_rays, _ = trajectory.shape
    lo, hi = (None, None) if clip is None else (np.asarray(clip[0]), np.asarray(clip[1]))

    vertices: list[np.ndarray] = []
    entities = []
    for j in range(n_rays):
        pts = trajectory[:, j, :]
        # Drop repeated points: a ray frozen after termination (lost / landed)
        # repeats its final position for every remaining step.
        poly = [pts[0]]
        for k in range(1, steps):
            if not np.allclose(pts[k], poly[-1]):
                poly.append(pts[k])
        if len(poly) < 2:
            continue  # nothing to draw, and no stray vertex left behind
        for run in _clipped_runs(poly, lo, hi):
            base = len(vertices)
            vertices.extend(run)
            entities.append(trimesh.path.entities.Line(list(range(base, base + len(run)))))

    if not entities:
        return scene
    path = trimesh.path.Path3D(entities=entities, vertices=np.asarray(vertices))
    _color_path(path, color)
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
