from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..core.intersections import intersect_plane
from ..core.ray_bundle import LazyRayBundle, RayBundle
from ..core.transforms import euler_to_matrix

if TYPE_CHECKING:
    from pathlib import Path

    from .detector import PhotoDetector
    from .optics import Concentrator
    from .sensor_group import SensorGroup


# Pipeline functions


def intersect_sensor(
    camera: Camera,
    ray_bundle: RayBundle | LazyRayBundle,
    sensor_idx: int = 0,
) -> tuple[RayBundle, Array]:
    """Intersect 3D rays with sensor planes.

    Accepts either a flat :class:`RayBundle` or a :class:`LazyRayBundle`.

    Returns:
        Tuple of ``(sensor_rays, s_idx)`` where *sensor_rays* is a
        :class:`~iactrace.core.ray_bundle.RayBundle` in the sensor-local
        frame and *s_idx* identifies which sensor each ray hit. A ray
        missing every sensor plane terminates: ``sensor_rays.alive`` is the
        incoming liveness AND-ed with "intersected a sensor".
    """
    if isinstance(ray_bundle, LazyRayBundle):
        ray_bundle = ray_bundle.materialise()
    sensor = camera.sensor_groups[sensor_idx]
    origins = ray_bundle.origins
    directions = ray_bundle.directions
    n_rays = origins.shape[0]

    def f(s_idx):
        # Calculate intersections with sensor plane
        pos = sensor.positions[s_idx]
        rot = euler_to_matrix(sensor.rotations[s_idx])
        pts, ts = jax.vmap(intersect_plane, in_axes=(0, 0, None, None))(
            origins, directions, pos, rot
        )
        # Check if intersection is within sensor bound
        ts = jnp.where(sensor.in_bounds(pts[:, 0], pts[:, 1]), ts, jnp.inf)
        local_dirs = directions @ rot
        return pts, ts, local_dirs

    # Map over all sensors in SensorGroup
    all_pts, all_ts, all_dirs = jax.vmap(f)(jnp.arange(sensor.n_sensors))

    # Get sensor id for minimum sensor and assign correctly
    s_idx = jnp.argmin(all_ts, axis=0).astype(jnp.int32)
    idx = jnp.arange(n_rays)
    pts = all_pts[s_idx, idx]
    t_sensor = all_ts[s_idx, idx]
    local_dirs = all_dirs[s_idx, idx]

    # A ray missing every sensor plane terminates (geometry loss); combine
    # with the liveness the bundle already carries so upstream-dead rays
    # (shadowed, off-aperture, absorbed geometry) stay dead here too.
    hit = jnp.isfinite(t_sensor)
    alive = ray_bundle.alive & hit
    path_length = ray_bundle.path_length + jnp.where(
        hit,
        t_sensor * ray_bundle.n,
        0.0,
    )

    sensor_rays = RayBundle(
        origins=jnp.stack([pts[:, 0], pts[:, 1], jnp.zeros(n_rays)], axis=-1),
        directions=local_dirs,
        values=jnp.where(alive, ray_bundle.values, 0.0),
        path_length=path_length,
        n=ray_bundle.n,
        alive=alive,
    )
    return sensor_rays, s_idx


def _run_chain(
    camera: Camera,
    ray_bundle: RayBundle,
    sensor_idx: int,
) -> tuple[Array, Array, RayBundle]:
    """Intersect the sensor, assign pixels, translate to pixel-local, run the chain.

    The pixel assignment (``pixel_index_and_mask``) is computed exactly once
    here and drives both the pixel-local reframing (``to_pixel_frame``) and
    the downstream binning / detection masks, so the frame a ray is traced in
    and the pixel its signal lands in can never disagree. Liveness travels on
    the bundle itself: ``pe_rays.alive`` is ``True`` only for rays that hit a
    sensor tile *and* landed on the photodetector surface.

    Returns ``(pix_id, valid, pe_rays)``; the image / matrix folds take
    ``pe_rays.values`` and ``collect`` reads the whole bundle.
    """
    sensor = camera.sensor_groups[sensor_idx]
    sensor_rays, s_idx = intersect_sensor(camera, ray_bundle, sensor_idx)
    x, y = sensor_rays.origins[:, 0], sensor_rays.origins[:, 1]
    pix_id, valid = sensor.pixel_index_and_mask(s_idx, x, y)
    local = sensor.to_pixel_frame(sensor_rays, pix_id)
    pe_rays = sensor.chain.propagate(local)
    return pix_id, valid, pe_rays


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
        camera.collect(rb)                # (pe_vals, pe_times, pix_id, detected)
        rb.materialise()                  # flat camera-frame RayBundle

    Attributes:
        sensor_groups: List of SensorGroup objects. Each group owns its pixel
            layout and its detection chain (optional concentrator + ``gap``
            + photodetector), so different groups can carry different cones or
            photodetectors. Configure a group's chain when constructing it, or via
            :meth:`set_concentrator` / :meth:`set_photodetector` / :meth:`set_gap`.
    """

    sensor_groups: list[SensorGroup]

    def __init__(self, sensor_groups: list[SensorGroup]) -> None:
        self.sensor_groups = list(sensor_groups)

    def _require_sensor_groups(self, sensor_idx: int) -> None:
        """Validate that ``sensor_idx`` references an existing sensor group."""
        if not self.sensor_groups:
            raise ValueError(
                "Camera has no sensor groups. Add a SensorGroup before calling collect()/image()."
            )
        if not 0 <= sensor_idx < len(self.sensor_groups):
            raise IndexError(
                f"sensor_idx={sensor_idx} out of range for "
                f"{len(self.sensor_groups)} sensor group(s)"
            )

    # Pipeline

    def collect(
        self,
        ray_bundle: RayBundle | LazyRayBundle,
        sensor_idx: int = 0,
    ) -> tuple[Array, Array, Array, Array]:
        """Per-ray output ``(pe_vals, pe_times, pix_id, detected)``.

        ``detected`` is the final liveness flag: the chain output's ``alive``
        (stayed alive through the optics, hit a sensor tile, landed on the
        photodetector surface) AND-ed with the pixel mask (inside a real pixel,
        outside the edge deadband). It is ``False`` for a ray lost anywhere
        along the way. Entries of ``pix_id`` / ``pe_times`` for undetected
        rays are meaningless and must be filtered with this mask before use;
        ``pe_vals`` is already zeroed there.

        Materialises a :class:`LazyRayBundle`: per-ray output cannot be
        produced incrementally.
        """
        self._require_sensor_groups(sensor_idx)
        if isinstance(ray_bundle, LazyRayBundle):
            ray_bundle = ray_bundle.materialise()

        pix_id, valid, pe_rays = _run_chain(self, ray_bundle, sensor_idx)
        detected = pe_rays.alive & valid
        pe_vals = jnp.where(detected, pe_rays.values, 0.0)
        return pe_vals, pe_rays.path_length, pix_id, detected

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

            def fold_element(image, rb_cam):
                pix_id, valid, pe_rays = _run_chain(self, rb_cam, sensor_idx)
                return image + sensor.scatter(pix_id, valid, pe_rays.values)

            return ray_bundle.fold(fold_element, init)

        pix_id, valid, pe_rays = _run_chain(self, ray_bundle, sensor_idx)
        return sensor.scatter(pix_id, valid, pe_rays.values)

    def response_matrix(
        self,
        lazy_bundle: LazyRayBundle,
        sensor_idx: int = 0,
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
        init = jnp.zeros((n_sources, sensor.n_sensors) + sensor.get_accumulator_shape())

        def fold_element(matrix, rb_cam):
            pix_id, valid, pe_rays = _run_chain(self, rb_cam, sensor_idx)

            # Per-element rays are source-major (see _build_source_rays):
            # the first n_samples rays belong to source 0, etc.
            def per_source(a):
                return a.reshape(n_sources, n_samples)

            contrib = jax.vmap(sensor.scatter)(
                per_source(pix_id),
                per_source(valid),
                per_source(pe_rays.values),
            )
            return matrix + contrib

        return lazy_bundle.fold(fold_element, init)

    # Composition
    #
    # Functional updates: every setter returns a new Camera (Equinox modules are
    # immutable pytrees), rebuilding only the affected sensor group / chain via
    # ``eqx.tree_at``.

    def _with_sensor_group(self, sensor_idx: int, new_group: SensorGroup) -> Camera:
        """Swap one sensor group into a copy of the camera."""
        new_groups = list(self.sensor_groups)
        new_groups[sensor_idx] = new_group
        return eqx.tree_at(lambda c: c.sensor_groups, self, new_groups)

    def set_sensor_positions(self, sensor_idx: int, positions: Array) -> Camera:
        """Set positions for sensors in a group."""
        group = self.sensor_groups[sensor_idx]
        new_group = eqx.tree_at(lambda s: s.positions, group, jnp.asarray(positions))
        return self._with_sensor_group(sensor_idx, new_group)

    def set_sensor_rotations(self, sensor_idx: int, rotations: Array) -> Camera:
        """Set rotations for sensors in a group."""
        group = self.sensor_groups[sensor_idx]
        new_group = eqx.tree_at(lambda s: s.rotations, group, jnp.asarray(rotations))
        return self._with_sensor_group(sensor_idx, new_group)

    def set_concentrator(self, sensor_idx: int, concentrator: Concentrator | None) -> Camera:
        """Set/replace the concentrator on sensor group ``sensor_idx``'s chain."""
        group = self.sensor_groups[sensor_idx]
        return self._with_sensor_group(sensor_idx, group.with_concentrator(concentrator))

    def set_photodetector(self, sensor_idx: int, photodetector: PhotoDetector) -> Camera:
        """Set/replace the photodetector on sensor group ``sensor_idx``'s chain."""
        group = self.sensor_groups[sensor_idx]
        return self._with_sensor_group(sensor_idx, group.with_photodetector(photodetector))

    def set_gap(self, sensor_idx: int, gap: float) -> Camera:
        """Set the gap (upstream exit -> detector spacing) on a group's chain."""
        group = self.sensor_groups[sensor_idx]
        return self._with_sensor_group(sensor_idx, group.with_gap(gap))

    def get_info(self) -> dict[str, Any]:
        """Summary of the camera configuration.

        Each sensor group reports its own geometry and detection chain, since the
        chain (concentrator + gap + photodetector) is owned per group.
        """
        info: dict[str, Any] = {"n_sensor_groups": len(self.sensor_groups)}
        for i, sg in enumerate(self.sensor_groups):
            chain = sg.chain
            group_info: dict[str, Any] = {
                "type": type(sg).__name__,
                "n_sensors": sg.n_sensors,
                "accumulator_shape": sg.get_accumulator_shape(),
                "photodetector_type": type(chain.photodetector).__name__,
                "has_concentrator": chain.concentrator is not None,
                "gap": float(chain.gap),
                "detector_z": float(chain.detector_z),
            }
            if chain.concentrator is not None:
                group_info["concentrator_type"] = type(chain.concentrator).__name__
            info[f"sensor_group_{i}"] = group_info
        return info

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
