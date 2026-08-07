from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..core.ray_bundle import LazyRayBundle
from ..core.render import apply_final_leg_shadow, final_leg_points, trace_optics
from ..core.trajectory import TraceResult, Trajectory
from ..core.transforms import euler_to_matrix
from . import operations as _ops

if TYPE_CHECKING:
    from pathlib import Path

    from ..core.obstructions import ObstructionGroup
    from ..core.optics import OpticalElementGroup


class Telescope(eqx.Module):
    """IACT telescope optical system as an Equinox Module.

    The Telescope owns the full optical path including the camera frame
    (position and orientation of the detector plane). After ray tracing,
    rays are transformed into the camera frame so that the Camera class
    operates purely in its local coordinate system.

    Optical elements (mirrors and lenses) are stored separately for clarity,
    but can be accessed together via the ``optical_groups`` property for
    unified ray tracing through mixed reflective/refractive systems.
    """

    mirror_groups: list[OpticalElementGroup]
    lens_groups: list[OpticalElementGroup]
    obstruction_groups: list[ObstructionGroup]
    camera_position: Array  # (3,) camera origin in world frame
    camera_rotation: Array  # (3,) Euler angles (degrees) for camera orientation
    name: str = eqx.field(static=True)

    def __init__(
        self,
        mirror_groups: list[OpticalElementGroup],
        obstruction_groups: list[ObstructionGroup] | None = None,
        name: str = "telescope",
        lens_groups: list[OpticalElementGroup] | None = None,
        camera_position: ArrayLike | None = None,
        camera_rotation: ArrayLike | None = None,
    ) -> None:
        """Initialize Telescope.

        Args:
            mirror_groups: List of Mirror groups (reflective elements)
            obstruction_groups: List of Obstruction groups
            name: Telescope name
            lens_groups: List of Lens groups (refractive elements)
            camera_position: Camera origin in world coordinates (3,).
                Defaults to [0, 0, 0].
            camera_rotation: Camera orientation as Euler angles in degrees (3,).
                Defaults to [0, 0, 0].
        """
        # Validate one group per optical stage
        all_groups = list(mirror_groups)
        if lens_groups:
            all_groups.extend(lens_groups)
        stages_seen: dict[int, str] = {}
        for g in all_groups:
            stage = g.optical_stage
            name_str = type(g).__name__
            if stage in stages_seen:
                raise ValueError(
                    f"Multiple groups at optical stage {stage}: "
                    f"{stages_seen[stage]} and {name_str}. "
                    f"Only one group per stage is allowed."
                )
            stages_seen[stage] = name_str

        self.mirror_groups = mirror_groups
        self.lens_groups = lens_groups if lens_groups else []
        self.obstruction_groups = obstruction_groups if obstruction_groups else []
        self.name = name

        if camera_position is None:
            camera_position = jnp.zeros(3)
        if camera_rotation is None:
            camera_rotation = jnp.zeros(3)
        self.camera_position = jnp.asarray(camera_position)
        self.camera_rotation = jnp.asarray(camera_rotation)

    @property
    def optical_groups(self) -> list[OpticalElementGroup]:
        """Return all optical groups (mirrors + lenses) combined."""
        return list(self.mirror_groups) + list(self.lens_groups)

    def render(
        self,
        sources: Array,
        values: Array,
        source_type: Literal["point", "parallel"] = "point",
    ) -> LazyRayBundle:
        """Describe a render through the optics; do not execute yet.

        Returns a :class:`LazyRayBundle` packaging this telescope's
        optics and camera frame with the given sources. Downstream
        camera methods (:meth:`Camera.image`,
        :meth:`Camera.response_matrix`, :meth:`Camera.collect`) consume
        it directly: image and response_matrix fold per primary element
        so the full ray buffer is never materialised; collect
        materialises (per-ray output cannot be folded).

        Args:
            sources: Source positions ``(N, 3)`` or directions ``(N, 3)``.
            values: Source strengths ``(N,)``. For ``'parallel'`` these are
                irradiances on the aperture; for ``'point'`` they are
                radiant intensities, and the irradiance each primary sample
                sees is ``value / d^2`` for that sample's distance ``d`` to
                the source.
            source_type: ``'point'`` or ``'parallel'``.
        """
        return LazyRayBundle(
            optical_groups=self.optical_groups,
            obstruction_groups=self.obstruction_groups,
            camera_position=self.camera_position,
            camera_rotation=self.camera_rotation,
            sources=sources,
            source_values=values,
            source_type=source_type,
        )

    def trace(
        self,
        ray_origins: Array,
        ray_directions: Array,
        values: Array,
        record_trajectory: bool = False,
    ) -> TraceResult:
        """Trace rays from arbitrary origins through the full optical system.

        Args:
            ray_origins: Ray starting positions (N, 3)
            ray_directions: Ray directions (N, 3), should be normalized
            values: Ray intensities (N,)
            record_trajectory: When True, also record the per-stage ray path for
                diagnostics / 3D visualization (see
                :func:`iactrace.viz.show_telescope`). Off by default and free when
                off -- nothing extra is computed. Mirrors the
                :func:`~iactrace.camera.trace_chain` option on the chain side.

        Returns:
            A :class:`~iactrace.core.trajectory.TraceResult`, as every tracer
            returns. Its ``rays`` are in the camera's local coordinate system;
            pass them to ``camera.collect()`` or ``camera.image()`` for
            detection::

                rays = telescope.trace(origins, directions, values).rays
                rays, trajectory = telescope.trace(..., record_trajectory=True)

            ``trajectory`` is ``None`` unless ``record_trajectory`` was set, in
            which case the :class:`~iactrace.core.trajectory.Trajectory` holds
            the source point, each optical stage's landing point, and finally
            the converging leg's landing on the camera reference plane --
            ``(n_stages + 2, N, 3)``, so the beam is seen coming to a focus.
            ``rays`` stops on the last optic (the sensor intersection happens
            downstream in ``Camera``); the trajectory adds that last leg for
            display.

            The trajectory is in the **world frame** (the frame
            :func:`iactrace.viz.show_telescope` draws the optics in), *not* the
            camera frame ``rays`` is reframed into.
        """
        result = trace_optics(
            self.optical_groups,
            self.obstruction_groups,
            ray_origins,
            ray_directions,
            values,
            record_trajectory=record_trajectory,
        )
        # Handoff = shadow the final leg (explicit), then a pure reframe.
        rb = apply_final_leg_shadow(
            result.rays,
            self.obstruction_groups,
            self.camera_position,
            self.camera_rotation,
        )
        if result.trajectory is None:
            return TraceResult(rb.to_frame(self.camera_position, self.camera_rotation))
        # Close the path on the focal plane: the bundle itself stops on the
        # last optic, so without this the beam is never seen converging.
        points = result.trajectory.points
        landing = final_leg_points(
            rb, self.camera_position, self.camera_rotation, fallback=points[-1]
        )
        trajectory = Trajectory(points=jnp.concatenate([points, landing[None]], axis=0))
        return TraceResult(rb.to_frame(self.camera_position, self.camera_rotation), trajectory)

    @classmethod
    def from_yaml(
        cls,
        filename: str | Path,
        n_samples: int = 100,
        *,
        key: Array,
    ) -> Telescope:
        """Load a telescope from a standalone telescope YAML file.

        The telescope file describes only the optical system and the
        camera frame. Load the camera separately via
        :meth:`Camera.from_yaml`.

        Args:
            filename: Path to telescope YAML file.
            n_samples: Number of Monte Carlo samples per mirror element.
            key: JAX random key for sampling and roughness.
        """
        from ..io.yaml_io import load_telescope_config

        return load_telescope_config(filename, n_samples, key=key)

    def to_yaml(
        self,
        filename: str | Path,
        precision: int = 6,
        overwrite: bool = True,
    ) -> Path:
        """Save the telescope configuration to a standalone YAML file.

        Args:
            filename: Output file path.
            precision: Number of decimal places for float values.
            overwrite: If True, overwrite existing file.

        Returns:
            Path to the saved file.
        """
        from ..io.yaml_io import save_telescope

        return save_telescope(self, filename, precision, overwrite)

    def to_dict(self) -> dict[str, Any]:
        """Convert telescope to a telescope-only configuration dictionary."""
        from ..io.yaml_io import telescope_to_dict

        return telescope_to_dict(self)

    # Camera-frame convenience methods

    def focus(self, delta_z: float) -> Telescope:
        """Adjust camera position along the camera's local z-axis (optical axis)."""
        rot = euler_to_matrix(self.camera_rotation)
        # Camera z-axis in world coordinates is the third column of rot
        new_position = self.camera_position + delta_z * rot[:, 2]
        return eqx.tree_at(lambda t: t.camera_position, self, new_position)

    def set_camera_position(self, position: Array) -> Telescope:
        """Set the camera position in world coordinates."""
        return eqx.tree_at(lambda t: t.camera_position, self, jnp.asarray(position))

    def set_camera_rotation(self, rotation: Array) -> Telescope:
        """Set the camera rotation (Euler angles in degrees)."""
        return eqx.tree_at(lambda t: t.camera_rotation, self, jnp.asarray(rotation))

    # Stage access

    def stage(self, idx: int) -> OpticalElementGroup:
        """Return the :class:`OpticalElementGroup` at ``optical_stage == idx``."""
        for g in self.optical_groups:
            if g.optical_stage == idx:
                return g
        available = sorted(g.optical_stage for g in self.optical_groups)
        raise IndexError(f"no stage {idx}; available: {available}")

    def stage_indices(self) -> list[int]:
        """Sorted list of all optical stages present."""
        return sorted(g.optical_stage for g in self.optical_groups)

    def stages_of_kind(self, kind: str) -> list[int]:
        """Sorted stages whose group has the given kind ('mirror', 'lens', 'slab')."""
        return sorted(g.optical_stage for g in self.optical_groups if g.kind == kind)

    @property
    def n_stages(self) -> int:
        return len(self.optical_groups)

    @property
    def n_mirror_elements(self) -> int:
        return sum(len(g) for g in self.optical_groups if g.kind == "mirror")

    @property
    def n_lens_elements(self) -> int:
        return sum(len(g) for g in self.optical_groups if g.kind in ("lens", "slab"))

    # Generic stage operations (any kind)

    def set_positions(self, stage: int, positions: Array) -> Telescope:
        return _ops.set_positions(self, stage, positions)

    def set_rotations(self, stage: int, rotations: Array) -> Telescope:
        return _ops.set_rotations(self, stage, rotations)

    def apply_displacement(self, stage: int, sigma_z: float, key: Array) -> Telescope:
        return _ops.apply_displacement(self, stage, sigma_z, key)

    def apply_misalignment(
        self, stage: int, sigma_h: float, sigma_v: float, key: Array
    ) -> Telescope:
        return _ops.apply_misalignment(self, stage, sigma_h, sigma_v, key)

    def apply_roughness(self, stage: int, sigma: float) -> Telescope:
        return _ops.apply_roughness(self, stage, sigma)

    def set_curvatures(self, stage: int, curvatures: Array) -> Telescope:
        return _ops.set_curvatures(self, stage, curvatures)

    def set_conics(self, stage: int, conics: Array) -> Telescope:
        return _ops.set_conics(self, stage, conics)

    def set_aspherics(self, stage: int, aspherics: Array) -> Telescope:
        return _ops.set_aspherics(self, stage, aspherics)

    def scale_curvatures(self, stage: int, factor: Array | float) -> Telescope:
        return _ops.scale_curvatures(self, stage, factor)

    def offset_curvatures(self, stage: int, offset: Array | float) -> Telescope:
        return _ops.offset_curvatures(self, stage, offset)

    def apply_conic_error(self, stage: int, sigma: float, key: Array) -> Telescope:
        return _ops.apply_conic_error(self, stage, sigma, key)

    def apply_aspheric_error(self, stage: int, sigmas: Array, key: Array) -> Telescope:
        return _ops.apply_aspheric_error(self, stage, sigmas, key)

    def apply_zernike_error(self, stage: int, sigmas: Array, key: Array) -> Telescope:
        return _ops.apply_zernike_error(self, stage, sigmas, key)

    def apply_astigmatism(self, stage: int, sigma: float, key: Array) -> Telescope:
        return _ops.apply_astigmatism(self, stage, sigma, key)

    def apply_coma(self, stage: int, sigma: float, key: Array) -> Telescope:
        return _ops.apply_coma(self, stage, sigma, key)

    def apply_trefoil(self, stage: int, sigma: float, key: Array) -> Telescope:
        return _ops.apply_trefoil(self, stage, sigma, key)

    def resample(self, stage: int, key: Array) -> Telescope:
        return _ops.resample(self, stage, key)

    # Kind-specific stage operations

    def set_reflectivity(self, stage: int, reflectivity: Array | float) -> Telescope:
        return _ops.set_reflectivity(self, stage, reflectivity)

    def scale_reflectivity(self, stage: int, factor: Array | float) -> Telescope:
        return _ops.scale_reflectivity(self, stage, factor)

    def set_transmittance(self, stage: int, transmittance: Array | float) -> Telescope:
        return _ops.set_transmittance(self, stage, transmittance)

    def scale_transmittance(self, stage: int, factor: Array | float) -> Telescope:
        return _ops.scale_transmittance(self, stage, factor)

    def set_refractive_index(self, stage: int, n_inside: Array | float) -> Telescope:
        return _ops.set_refractive_index(self, stage, n_inside)

    def set_thickness(self, stage: int, thickness: Array | float) -> Telescope:
        return _ops.set_thickness(self, stage, thickness)

    def set_focal_lengths(
        self, stage: int, focal_lengths: Array, n_outside: float = 1.0
    ) -> Telescope:
        return _ops.set_focal_lengths(self, stage, focal_lengths, n_outside)

    def apply_focal_error(
        self,
        stage: int,
        sigma: float,
        key: Array,
        relative: bool = False,
        n_outside: float = 1.0,
    ) -> Telescope:
        return _ops.apply_focal_error(self, stage, sigma, key, relative, n_outside)

    # Obstruction methods

    def add_obstruction(self, obstruction: ObstructionGroup) -> Telescope:
        return _ops.add_obstruction(self, obstruction)

    def remove_obstruction(self, group_idx: int) -> Telescope:
        return _ops.remove_obstruction(self, group_idx)

    def clear_obstructions(self) -> Telescope:
        return _ops.clear_obstructions(self)

    def get_obstruction_count(self) -> int:
        return _ops.get_obstruction_count(self)

    def get_info(self) -> dict[str, Any]:
        return _ops.get_info(self)
