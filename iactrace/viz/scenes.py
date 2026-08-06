import numpy as np
import trimesh

from ..core import euler_to_matrix
from ._meshes import _apply_color, _replicate_mesh, _rigid_transform, _spin_z
from ._optics import _get_lens_meshes, _get_mirror_meshes, _get_obstruction_meshes
from ._sensors import (
    _chain_part_meshes,
    _get_sensor_grid_paths,
    _get_sensor_meshes,
    _pixel_centers_2d,
)
from .overlays import add_trajectories


def show_telescope(telescope, camera=None, *, trajectory=None, **kwargs):
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
        trajectory: Optional recorded ray paths to draw as polylines, in the
            world frame (the frame the optics are drawn in). A ``(steps + 1, N, 3)``
            array as returned by ``Telescope.trace(..., record_trajectory=True)``.
            Trace yourself and pass the result -- this function does not run the
            tracer. Drawn via :func:`add_trajectories`.
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

    # Optional recorded ray paths (traced by the caller), drawn as-is.
    if trajectory is not None:
        add_trajectories(scene, trajectory, color=kwargs.get("trace_color", [0, 0, 255, 255]))

    return scene


def show_camera(
    camera,
    *,
    sensor_idx=None,
    trajectory=None,
    clip=True,
    clip_margin=0.25,
    entrances=False,
    detail=0.5,
    **kwargs,
):
    """Visualize the whole camera in 3D: every pixel's concentrator and photosensor.

    Replicates each sensor group's detection chain -- the same geometry
    :func:`show_sensor_chain` draws for one pixel (concentrator walls,
    photocathode, photodetector body) -- at every pixel of every sensor in the
    group, placed by the sensor's own position and Euler rotation. The result is
    the camera as the camera file describes it, in the camera frame.

    Each part is emitted as one merged mesh per sensor group, so a few thousand
    pixels stay a handful of geometries rather than a few thousand.

    For a single pixel's chain in isolation (and for overlaying traced rays on
    it), use :func:`show_sensor_chain`. For a *camera image* -- pixel values on
    the focal plane -- use :func:`iactrace.viz.show_image`.

    Recorded ray paths can be overlaid on the geometry -- the final leg onto the
    camera joined to the scattering through each pixel's chain::

        show_camera(camera, trajectory=camera.trace(rays))

    Args:
        camera: Camera object; every sensor group is drawn unless ``sensor_idx``
            selects one.
        sensor_idx: Optional index of a single sensor group to draw.
        trajectory: Optional camera-frame ray paths to draw as polylines, as
            returned by :meth:`Camera.trace`. As elsewhere, this function draws
            what it is given and never runs the tracer itself.
        clip: How to trim those paths. ``True`` (default) fits a box around the
            drawn camera geometry, padded by ``clip_margin``; ``False`` draws
            them whole; or pass an explicit ``(lo, hi)`` box, each ``(3,)``, in
            the camera frame. Only the paths are clipped -- the geometry is
            always drawn in full.
        clip_margin: Padding of the automatic box, as a fraction of the camera's
            larger transverse extent (default ``0.25``). This is what sets how
            much of the incoming beam you see converging above the pixels; raise
            it to show more of the approach, lower it to sit tight on the camera.
        entrances: Also draw each pixel's entrance-aperture face (default
            ``False`` -- with thousands of pixels the filled faces hide the
            cones behind them).
        detail: Tessellation scale, ``1.0`` being the full per-pixel detail
            :func:`show_sensor_chain` uses. Defaults to ``0.5``: the whole-camera
            mesh is the per-pixel one times a few thousand, and at camera scale
            the extra facets are invisible. Raise it when zooming in.
        **kwargs: ``cone_color`` / ``detector_color`` / ``sensor_color`` /
            ``entrance_color`` RGBA, as in :func:`show_sensor_chain`.

    Returns:
        trimesh.Scene
    """
    groups = camera.sensor_groups
    if sensor_idx is not None:
        groups = [groups[sensor_idx]]

    scene = trimesh.Scene()
    for sensor in groups:
        parts = _chain_part_meshes(sensor, include_entrance=entrances, detail=detail, **kwargs)
        if not parts:
            continue

        centers_2d = _pixel_centers_2d(sensor)
        # Pixel centres sit in the sensor plane (local z = 0), which is where the
        # chain's entrance aperture is, so the chain hangs off it along local -z.
        local = np.column_stack([centers_2d, np.zeros(len(centers_2d))])
        positions = np.asarray(sensor.positions)
        rotations = np.asarray(sensor.rotations)
        spin = _spin_z(sensor.pixel_frame_rotation)

        for mesh, color in parts:
            merged = []
            for i in range(len(sensor)):
                rot = np.asarray(euler_to_matrix(rotations[i]))
                offsets = local @ rot.T + positions[i]
                merged.append(_replicate_mesh(mesh, rot @ spin, offsets))
            combined = merged[0] if len(merged) == 1 else trimesh.util.concatenate(merged)
            _apply_color(combined, color)
            scene.add_geometry(combined)

    # Optional recorded ray paths (traced by the caller), drawn as-is.
    if trajectory is not None:
        add_trajectories(
            scene,
            trajectory,
            color=kwargs.get("trace_color", [0, 0, 255, 255]),
            clip=_camera_clip_box(scene, clip, clip_margin),
        )

    return scene


def _camera_clip_box(scene, clip, margin):
    """Box to trim ray paths to, from the camera geometry already in ``scene``.
    ``clip`` is ``True`` for an automatic box, ``False`` / ``None`` for no
    clipping, or an explicit ``(lo, hi)`` passed straight through.
    The automatic box is the drawn geometry's own bounds grown by ``margin``
    times the larger transverse extent. Growing it in every direction, not just
    along the axis the beam arrives on, keeps rays converging steeply toward an
    edge pixel: at the top of the box those are still outside the pixel field.
    """
    if clip is None or clip is False:
        return None
    if clip is not True:
        return clip
    if not scene.geometry:
        return None
    lo, hi = np.asarray(scene.bounds, dtype=float)
    pad = margin * max(hi[0] - lo[0], hi[1] - lo[1])
    return lo - pad, hi + pad


def show_sensor_chain(camera, sensor_idx=0, *, trajectory=None, **kwargs):
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

    Ray paths are *not* traced here -- run the chain tracer yourself and pass the
    recorded ``trajectory`` (pixel-local frame) to overlay it, e.g.::

        trace = trace_chain(cone, chain.surface, rays, record_trajectory=True)
        show_sensor_chain(camera, trajectory=trace)  # ChainTrace or its array

    Args:
        camera: Camera object (the selected sensor group and its ``chain`` are read).
        sensor_idx: Which sensor group to take the pixel geometry and chain from.
        trajectory: Optional recorded ray paths to draw as polylines, in the
            pixel-local frame: a ``(steps + 1, N, 3)`` array or any object exposing
            a ``trajectory`` attribute (e.g. a
            :class:`~iactrace.camera.optics.ChainTrace`). Drawn via
            :func:`add_trajectories`.
        **kwargs: ``entrance_color`` / ``cone_color`` / ``detector_color`` /
            ``sensor_color`` / ``trace_color`` RGBA.

    Returns:
        trimesh.Scene
    """
    sensor = camera.sensor_groups[sensor_idx]
    scene = trimesh.Scene()

    for mesh, color in _chain_part_meshes(sensor, **kwargs):
        _apply_color(mesh, color)
        scene.add_geometry(mesh)

    # Optional recorded ray paths (traced by the caller), drawn as-is.
    if trajectory is not None:
        add_trajectories(scene, trajectory, color=kwargs.get("trace_color", [0, 0, 255, 255]))

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
