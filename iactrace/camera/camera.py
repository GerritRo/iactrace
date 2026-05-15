from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.intersections import intersect_plane
from ..core.ray_bundle import LazyRayBundle, RayBundle
from ..core.transforms import euler_to_matrix
from . import operations as _ops
from .photosensor import PhotoSensor, UniformQE

if TYPE_CHECKING:
    from pathlib import Path

    from .concentrator import Concentrator
    from .layout import SensorGroup


# Pipeline functions


def intersect_sensor(
    camera: Camera,
    ray_bundle: RayBundle | LazyRayBundle,
    sensor_idx: int = 0,
) -> tuple[RayBundle, Array, Array]:
    """Intersect 3D rays with sensor planes.

    Accepts either a flat :class:`RayBundle` or a :class:`LazyRayBundle`
    (which is materialised first — per-ray output cannot be folded).

    Returns:
        Tuple of ``(sensor_rays, s_idx, hit_mask)`` where *sensor_rays*
        is a :class:`~iactrace.core.ray_bundle.RayBundle` in the
        sensor-local frame, *s_idx* identifies which sensor each ray hit,
        and *hit_mask* is True for rays that intersected a sensor.
    """
    if isinstance(ray_bundle, LazyRayBundle):
        ray_bundle = ray_bundle.materialise()
    sensor = camera.sensor_groups[sensor_idx]
    origins = ray_bundle.origins
    directions = ray_bundle.directions
    n_rays = origins.shape[0]

    def f(s_idx):
        pos = sensor.positions[s_idx]
        rot = euler_to_matrix(sensor.rotations[s_idx])
        pts, ts = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
            origins, directions, pos, rot)
        # Tiles are bounded — mask hits that fall outside this tile's
        # active footprint so argmin doesn't pick an infinite plane that
        # the ray crosses before reaching its true tile.
        in_b = sensor.in_bounds(pts[:, 0], pts[:, 1])
        ts = jnp.where(in_b, ts, jnp.inf)
        dx = directions @ rot[:, 0]
        dy = directions @ rot[:, 1]
        return pts, ts, dx, dy

    all_pts, all_ts, all_dx, all_dy = jax.vmap(f)(jnp.arange(sensor.n_sensors))
    s_idx = jnp.argmin(all_ts, axis=0)
    idx = jnp.arange(n_rays)
    pts = all_pts[s_idx, idx]
    t_sensor = all_ts[s_idx, idx]
    dx = all_dx[s_idx, idx]
    dy = all_dy[s_idx, idx]
    s_idx = s_idx.astype(jnp.int32)

    hit_mask = t_sensor < 1e10
    # Weight the final geometric leg by the ray's current medium index.
    path_length = ray_bundle.path_length + jnp.where(
        hit_mask, t_sensor * ray_bundle.n, 0.0,
    )
    dz = jnp.sqrt(jnp.maximum(1.0 - dx**2 - dy**2, 0.0))

    sensor_rays = RayBundle(
        origins=jnp.stack([pts[:, 0], pts[:, 1], jnp.zeros(n_rays)], axis=-1),
        directions=jnp.stack([dx, dy, dz], axis=-1),
        values=ray_bundle.values,
        path_length=path_length,
        n=ray_bundle.n,
    )
    return sensor_rays, s_idx, hit_mask


def apply_concentrator(camera: Camera, sensor_rays: RayBundle) -> RayBundle:
    """Apply concentrator to sensor-local rays if present."""
    if camera.concentrator is None:
        return sensor_rays
    return camera.concentrator.apply(sensor_rays)


def _project_to_sensor(
    camera: Camera, rb_cam: RayBundle, sensor_idx: int,
) -> tuple[Array, Array, Array, Array]:
    """Camera-frame rays through sensor plane + concentrator + photosensor.

    Returns ``(s_idx, x, y, pe_vals)`` ready for :meth:`SensorGroup.accumulate`.
    """
    sensor_rays, s_idx, _ = intersect_sensor(camera, rb_cam, sensor_idx)
    x, y = sensor_rays.origins[:, 0], sensor_rays.origins[:, 1]
    sensor_rays = apply_concentrator(camera, sensor_rays)
    pe = camera.photosensor.apply(sensor_rays.values)
    return s_idx, x, y, pe


# Camera


class Camera(eqx.Module):
    """Camera for photon collection and imaging.

    The Camera works in its local coordinate system.  Sensor positions
    and rotations are relative to the camera origin (typically [0, 0, 0]
    for a single-sensor camera).  The Telescope transforms the RayBundle
    into the camera frame before passing it here.

    Pipeline::

        rb = telescope.render(...)        # LazyRayBundle
        camera.image(rb)                  # pixel image (fused, per-element fold)
        camera.response_matrix(rb)        # per-source pixel response (fused)
        camera.collect(rb)                # (pe_vals, pe_times, pix_id, hit_mask)
                                          # — materialises rays
        rb.materialise()                  # flat camera-frame RayBundle

    Attributes:
        sensor_groups: List of SensorGroup objects defining pixel layout
        concentrator: Optional light concentrator (Winston cone, etc.)
        photosensor: Photosensor response model (quantum efficiency)
    """

    sensor_groups: list[SensorGroup]
    concentrator: Concentrator | None
    photosensor: PhotoSensor

    def __init__(
        self,
        sensor_groups: list[SensorGroup],
        concentrator: Concentrator | None = None,
        photosensor: PhotoSensor | None = None,
    ) -> None:
        self.sensor_groups = list(sensor_groups)
        self.concentrator = concentrator

        if photosensor is not None:
            self.photosensor = photosensor
        else:
            self.photosensor = UniformQE(1.0)

    def _require_sensor_groups(self, sensor_idx: int) -> None:
        """Validate that ``sensor_idx`` references an existing sensor group."""
        if not self.sensor_groups:
            raise ValueError(
                "Camera has no sensor groups. Add a SensorGroup before "
                "calling collect()/image()."
            )
        if not 0 <= sensor_idx < len(self.sensor_groups):
            raise IndexError(
                f"sensor_idx={sensor_idx} out of range for "
                f"{len(self.sensor_groups)} sensor group(s)"
            )

    # Pipeline

    def image(
        self,
        ray_bundle: RayBundle | LazyRayBundle,
        sensor_idx: int = 0,
    ) -> Array:
        """Pixel image of shape ``(n_sensors, *pixel_shape)``.

        Accepts either a flat :class:`RayBundle` (e.g. from
        :meth:`Telescope.trace`) or the :class:`LazyRayBundle` returned
        by :meth:`Telescope.render`. The lazy form folds per
        primary-mirror element so the full ray buffer is never
        materialised; the eager form scatters the buffer in one call.
        """
        self._require_sensor_groups(sensor_idx)
        sensor = self.sensor_groups[sensor_idx]

        if isinstance(ray_bundle, LazyRayBundle):
            init = jnp.zeros((sensor.n_sensors,) + sensor.get_accumulator_shape())

            def accumulate(image, rb_cam):
                s_idx, x, y, pe = _project_to_sensor(self, rb_cam, sensor_idx)
                return image + sensor.accumulate(s_idx, x, y, pe)

            return ray_bundle.fold(accumulate, init)

        s_idx, x, y, pe = _project_to_sensor(self, ray_bundle, sensor_idx)
        return sensor.accumulate(s_idx, x, y, pe)

    def collect(
        self,
        ray_bundle: RayBundle | LazyRayBundle,
        sensor_idx: int = 0,
    ) -> tuple[Array, Array, Array, Array]:
        """Per-ray output ``(pe_vals, pe_times, pix_id, hit_mask)``.

        ``hit_mask`` is ``True`` for rays that hit a sensor plane and
        ``False`` for rays that missed every sensor; entries in
        ``pix_id``/``pe_times`` for missed rays are meaningless and
        should be filtered with the mask before use.

        Materialises a :class:`LazyRayBundle`: per-ray output cannot be
        produced incrementally.
        """
        self._require_sensor_groups(sensor_idx)
        if isinstance(ray_bundle, LazyRayBundle):
            ray_bundle = ray_bundle.materialise()
        sensor = self.sensor_groups[sensor_idx]

        sensor_rays, s_idx, hit_mask = intersect_sensor(
            self, ray_bundle, sensor_idx,
        )
        pix_id = sensor.assign_pixels(
            s_idx, sensor_rays.origins[:, 0], sensor_rays.origins[:, 1],
        )
        sensor_rays = apply_concentrator(self, sensor_rays)
        pe_vals = self.photosensor.apply(sensor_rays.values)
        return pe_vals, sensor_rays.path_length, pix_id, hit_mask

    def response_matrix(
        self, lazy_bundle: LazyRayBundle, sensor_idx: int = 0,
    ) -> Array:
        """Per-source pixel response, shape ``(n_sources, n_sensors, *pixel_shape)``.

        Folds per stage-0 element instead of materialising the full ray
        buffer; peak memory is bounded by the matrix itself.

        Requires a :class:`LazyRayBundle` so the per-source structure
        is known. Pass ``telescope.render(...)`` directly.
        """
        if not isinstance(lazy_bundle, LazyRayBundle):
            raise TypeError(
                "response_matrix requires a LazyRayBundle from "
                "Telescope.render (per-source structure must be known); "
                f"got {type(lazy_bundle).__name__}."
            )
        self._require_sensor_groups(sensor_idx)

        stage0 = next(
            (g for g in lazy_bundle.optical_groups if g.optical_stage == 0),
            None,
        )
        if stage0 is None:
            raise ValueError("response_matrix requires a stage-0 optical group.")
        n_samples = stage0.n_samples
        n_sources = lazy_bundle.sources.shape[0]

        sensor = self.sensor_groups[sensor_idx]
        init = jnp.zeros(
            (n_sources, sensor.n_sensors) + sensor.get_accumulator_shape()
        )

        def accumulate(matrix, rb_cam):
            s_idx, x, y, pe = _project_to_sensor(self, rb_cam, sensor_idx)
            # Per-element rays are source-major (see _build_source_rays):
            # the first n_samples rays belong to source 0, etc.
            def per_source(a):
                return a.reshape(n_sources, n_samples)
            contrib = jax.vmap(sensor.accumulate)(
                per_source(s_idx), per_source(x), per_source(y), per_source(pe),
            )
            return matrix + contrib

        return lazy_bundle.fold(accumulate, init)

    # Composition

    def set_sensor_positions(self, sensor_idx: int, positions: Array) -> Camera:
        return _ops.set_sensor_positions(self, sensor_idx, positions)

    def set_sensor_rotations(self, sensor_idx: int, rotations: Array) -> Camera:
        return _ops.set_sensor_rotations(self, sensor_idx, rotations)

    def set_concentrator(self, concentrator: Concentrator | None) -> Camera:
        return _ops.set_concentrator(self, concentrator)

    def set_photosensor(self, photosensor: PhotoSensor) -> Camera:
        return _ops.set_photosensor(self, photosensor)

    def get_info(self) -> dict[str, Any]:
        return _ops.get_info(self)

    # I/O

    @classmethod
    def from_yaml(cls, filename: str | Path) -> Camera:
        """Load a Camera from a standalone camera YAML file.

        Sensor positions are interpreted as camera-local coordinates.

        Args:
            filename: Path to camera YAML file.

        Returns:
            Camera object.
        """
        from ..io.yaml_io import load_camera_config

        return load_camera_config(filename)

    def to_yaml(
        self,
        filename: str | Path,
        precision: int = 6,
        overwrite: bool = True,
    ) -> Path:
        """Save camera to a standalone YAML file.

        Sensor positions are written in camera-local coordinates.

        Args:
            filename: Output file path.
            precision: Number of decimal places for float values.
            overwrite: If True, overwrite existing file.

        Returns:
            Path to the saved file.
        """
        from ..io.yaml_io import save_camera

        return save_camera(self, filename, precision, overwrite)

    def to_dict(self) -> dict[str, Any]:
        """Convert camera to a standalone configuration dictionary.

        Sensor positions are in camera-local coordinates.
        """
        from ..io.yaml_io import camera_to_dict

        return camera_to_dict(self)