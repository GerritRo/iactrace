from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
from jax import Array

from ..core.render import render, render_debug, trace_rays, trace_rays_debug

if TYPE_CHECKING:
    from pathlib import Path

    from ..core import Integrator, ObstructionGroup, OpticalGroupBase
    from ..sensors import SensorGroup
    from .lenses import LensGroup
    from .mirrors import MirrorGroup


class Telescope(eqx.Module):
    """
    IACT telescope configuration as an Equinox Module.

    Stores mirrors, lenses, obstructions, and sensors as object lists for
    polymorphic dispatch while maintaining JAX compatibility.

    Optical elements (mirrors and lenses) are stored separately for clarity,
    but can be accessed together via the `optical_groups` property for unified
    ray tracing through mixed reflective/refractive systems.
    """

    mirror_groups: list[MirrorGroup]
    lens_groups: list[LensGroup] | None
    obstruction_groups: list[ObstructionGroup] | None
    sensors: list[SensorGroup]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        mirror_groups: list[MirrorGroup],
        obstruction_groups: list[ObstructionGroup] | None = None,
        sensors: list[SensorGroup] | None = None,
        name: str = "telescope",
        lens_groups: list[LensGroup] | None = None,
    ) -> None:
        """
        Initialize Telescope.

        Args:
            mirror_groups: List of Mirror groups (reflective elements)
            obstruction_groups: List of Obstruction groups
            sensors: List of sensor objects
            name: Telescope name
            lens_groups: List of Lens groups (refractive elements)
        """
        self.mirror_groups = mirror_groups
        self.lens_groups = lens_groups
        self.obstruction_groups = obstruction_groups
        self.sensors = list(sensors) if sensors else []
        self.name = name

    @property
    def optical_groups(self) -> list[OpticalGroupBase]:
        """Return all optical groups (mirrors + lenses) combined.

        This property provides a unified view of all optical elements for
        ray tracing through mixed reflective/refractive systems.
        """
        groups: list[OpticalGroupBase] = list(self.mirror_groups)
        if self.lens_groups:
            groups.extend(self.lens_groups)
        return groups

    def render(
        self,
        sources: Array,
        values: Array,
        source_type: Literal["point", "parallel"] = "point",
        sensor_idx: int = 0,
        debug: bool = False,
    ) -> Array | tuple[Array, Array]:
        """
        Render sources through telescope.

        Args:
            sources: Source positions (N, 3) or directions (N, 3)
            values: Flux values (N,)
            source_type: 'point' or 'parallel'
            sensor_idx: Which sensor to use
            debug: If True, return raw hits instead of accumulated image

        Returns:
            Rendered image or (pts, values) if debug=True

        Note:
            Shadowing is automatically applied if obstruction_groups is non-empty.
            Use telescope.clear_obstructions() to render without shadowing.
        """
        if debug:
            return render_debug(self, sources, values, source_type, sensor_idx)
        return render(self, sources, values, source_type, sensor_idx)

    def trace(
        self,
        ray_origins: Array,
        ray_directions: Array,
        values: Array,
        sensor_idx: int = 0,
        debug: bool = False,
    ) -> Array | tuple[Array, Array]:
        """
        Trace classical rays through the full optical system.

        Unlike __call__ which samples rays from primary mirror surfaces, this method
        traces rays from arbitrary external origins through the full optical system,
        including intersection with primary mirrors (stage 0). Per-mirror reflectivity
        is applied at each surface.

        Args:
            ray_origins: Ray starting positions (N, 3)
            ray_directions: Ray directions (N, 3), should be normalized
            values: Ray intensities (N,)
            sensor_idx: Which sensor to use
            debug: If True, return raw hits instead of accumulated image

        Returns:
            Rendered image or (pts, values) if debug=True

        Note:
            Shadowing is automatically applied if obstruction_groups is non-empty.
        """
        if debug:
            return trace_rays_debug(self, ray_origins, ray_directions, values, sensor_idx)
        return trace_rays(self, ray_origins, ray_directions, values, sensor_idx)

    @classmethod
    def from_yaml(
        cls,
        filename: str | Path,
        integrator: Integrator,
        key: Array | None = None,
    ) -> Telescope:
        """Load from YAML config."""
        from ..io.yaml_io import load_telescope

        return load_telescope(filename, integrator, key)

    def to_yaml(
        self,
        filename: str | Path,
        precision: int = 6,
        overwrite: bool = True,
    ) -> Path:
        """Save telescope configuration to a YAML file.

        This is the reverse of from_yaml() - it extracts the telescope configuration
        and saves it in a format that can be loaded back with from_yaml().

        Note:
            Some runtime state (sampling, perturbation) is not preserved in the YAML.
            When reloading, you'll need to provide an integrator to resample.

        Args:
            filename: Output file path.
            precision: Number of decimal places for float values.
            overwrite: If True, overwrite existing file. If False, raise
                      FileExistsError if file exists.

        Returns:
            Path to the saved file.

        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        from ..io.yaml_io import save_telescope

        return save_telescope(self, filename, precision, overwrite)

    def to_dict(self) -> dict[str, Any]:
        """Convert telescope to a configuration dictionary.

        Returns a dictionary in the same format as YAML configuration files,
        suitable for serialization or inspection.

        Returns:
            Configuration dictionary with telescope, mirror_templates,
            mirrors, obstructions, and sensors sections.
        """
        from ..io.yaml_io import telescope_to_dict

        return telescope_to_dict(self)

    # Convenience methods

    def resample_mirrors(self, integrator: Integrator, key: Array) -> Telescope:
        """Resample all mirrors with specified integrator.

        Args:
            integrator: Integrator object
            key: jax.random.key

        Returns:
            New telescope with resampled mirrors
        """
        from .operations import resample_mirrors

        return resample_mirrors(self, integrator, key)

    def set_mirror_positions(self, group_idx: int, positions: Array) -> Telescope:
        """Set positions for all mirrors in a group.

        Args:
            group_idx: Index of mirror group
            positions: New positions array (N, 3)

        Returns:
            New Telescope with updated mirror positions
        """
        from .operations import set_mirror_positions

        return set_mirror_positions(self, group_idx, positions)

    def set_mirror_rotations(self, group_idx: int, rotations: Array) -> Telescope:
        """Set rotations for all mirrors in a group.

        Args:
            group_idx: Index of mirror group
            rotations: New rotations array (N, 3) in degrees (Euler angles)

        Returns:
            New Telescope with updated mirror rotations
        """
        from .operations import set_mirror_rotations

        return set_mirror_rotations(self, group_idx, rotations)

    def scale_mirror_weights(
        self, group_idx: int, scale_factors: Array | float
    ) -> Telescope:
        """Scale reflectivity for mirrors in a group.

        Args:
            group_idx: Index of mirror group
            scale_factors: Scale factors per mirror (N,) or single value

        Returns:
            New Telescope with scaled mirror reflectivity
        """
        from .operations import scale_mirror_weights

        return scale_mirror_weights(self, group_idx, scale_factors)

    def apply_roughness(self, roughness: float) -> Telescope:
        """Apply roughness to all telescope mirrors.

        Args:
            roughness: Surface roughness in arcseconds

        Returns:
            New Telescope with updated roughness for mirrors
        """
        from .operations import apply_roughness

        return apply_roughness(self, roughness)

    def apply_roughness_to_group(self, group_idx: int, roughness: float) -> Telescope:
        """Apply roughness to a specific mirror group.

        Args:
            group_idx: Index of mirror group
            roughness: Surface roughness in arcseconds

        Returns:
            New Telescope with updated roughness for specified group
        """
        from .operations import apply_roughness_to_group

        return apply_roughness_to_group(self, group_idx, roughness)

    def apply_misalignment_to_group(
        self, group_idx: int, sigma_h: float, sigma_v: float, key: Array
    ) -> Telescope:
        """Apply random Gaussian misalignment to mirror orientations.

        Adds random perturbations to the horizontal and vertical
        angles of each mirror in the specified group, drawn from independent
        Gaussian distributions.

        Args:
            group_idx: Index of mirror group to modify
            sigma_h: Standard deviation of horizontal misalignment in arcseconds
            sigma_v: Standard deviation of vertical misalignment in arcseconds
            key: JAX random key for reproducibility

        Returns:
            New Telescope with randomly misaligned mirrors
        """
        from .operations import apply_misalignment_to_group

        return apply_misalignment_to_group(self, group_idx, sigma_h, sigma_v, key)

    def apply_displacement_to_group(
        self, group_idx: int, sigma_z: float, key: Array
    ) -> Telescope:
        """Apply random Gaussian displacement to mirrors along the z-axis.

        Adds random perturbations to the z-coordinate of each mirror position
        in the specified group, drawn from a Gaussian distribution.

        Args:
            group_idx: Index of mirror group to modify
            sigma_z: Standard deviation of z-axis displacement (same units as positions)
            key: JAX random key for reproducibility

        Returns:
            New Telescope with randomly displaced mirrors
        """
        from .operations import apply_displacement_to_group

        return apply_displacement_to_group(self, group_idx, sigma_z, key)

    def apply_focal_error_to_group(
        self,
        group_idx: int,
        sigma: float,
        key: Array,
        relative: bool = False,
    ) -> Telescope:
        """Apply random Gaussian error to mirror focal lengths.

        Perturbs the focal length of each mirror and converts back to curvature.
        For spherical/parabolic mirrors: f = 1/(2c), c = 1/(2f).

        Args:
            group_idx: Index of mirror group to modify
            sigma: Error magnitude:
                - If relative=True: fractional error (e.g., 0.01 for 1%)
                - If relative=False: absolute error in same units as focal length
            key: JAX random key for reproducibility
            relative: If True, apply relative (percentage) error; if False, absolute

        Returns:
            New Telescope with perturbed mirror curvatures
        """
        from .operations import apply_focal_error_to_group

        return apply_focal_error_to_group(self, group_idx, sigma, key, relative)

    def apply_conic_error_to_group(
        self,
        group_idx: int,
        sigma: float,
        key: Array,
    ) -> Telescope:
        """Apply random Gaussian error to mirror conic constants.

        Args:
            group_idx: Index of mirror group to modify
            sigma: Standard deviation of conic constant error
            key: JAX random key for reproducibility

        Returns:
            New Telescope with perturbed mirror conic constants
        """
        from .operations import apply_conic_error_to_group

        return apply_conic_error_to_group(self, group_idx, sigma, key)

    def apply_aspheric_error_to_group(
        self,
        group_idx: int,
        sigmas: Array,
        key: Array,
    ) -> Telescope:
        """Apply random Gaussian errors to mirror aspheric coefficients.

        Each aspheric term can have its own sigma value.

        Args:
            group_idx: Index of mirror group to modify
            sigmas: Standard deviations per aspheric term (K,). If fewer sigmas
                than terms, remaining terms get zero error.
            key: JAX random key for reproducibility

        Returns:
            New Telescope with perturbed mirror aspheric coefficients
        """
        from .operations import apply_aspheric_error_to_group

        return apply_aspheric_error_to_group(self, group_idx, sigmas, key)

    def set_mirror_curvatures(
        self,
        group_idx: int,
        curvatures: Array,
    ) -> Telescope:
        """Set curvatures for all mirrors in a group.

        Args:
            group_idx: Index of mirror group
            curvatures: New curvatures array (N,)

        Returns:
            New Telescope with updated mirror curvatures
        """
        from .operations import set_mirror_curvatures

        return set_mirror_curvatures(self, group_idx, curvatures)

    def set_mirror_conics(
        self,
        group_idx: int,
        conics: Array,
    ) -> Telescope:
        """Set conic constants for all mirrors in a group.

        Args:
            group_idx: Index of mirror group
            conics: New conic constants array (N,)

        Returns:
            New Telescope with updated mirror conic constants
        """
        from .operations import set_mirror_conics

        return set_mirror_conics(self, group_idx, conics)

    def set_mirror_aspherics(
        self,
        group_idx: int,
        aspherics: Array,
    ) -> Telescope:
        """Set aspheric coefficients for all mirrors in a group.

        Args:
            group_idx: Index of mirror group
            aspherics: New aspheric coefficients array (N, K) where K is number of terms

        Returns:
            New Telescope with updated mirror aspheric coefficients
        """
        from .operations import set_mirror_aspherics

        return set_mirror_aspherics(self, group_idx, aspherics)

    def scale_mirror_curvatures(
        self,
        group_idx: int,
        scale_factors: Array | float,
    ) -> Telescope:
        """Scale curvatures for mirrors in a group.

        Args:
            group_idx: Index of mirror group
            scale_factors: Scale factors per mirror (N,) or single value

        Returns:
            New Telescope with scaled mirror curvatures
        """
        from .operations import scale_mirror_curvatures

        return scale_mirror_curvatures(self, group_idx, scale_factors)

    def offset_mirror_curvatures(
        self,
        group_idx: int,
        offsets: Array | float,
    ) -> Telescope:
        """Add offset to curvatures for mirrors in a group.

        Args:
            group_idx: Index of mirror group
            offsets: Offsets per mirror (N,) or single value to add

        Returns:
            New Telescope with offset mirror curvatures
        """
        from .operations import offset_mirror_curvatures

        return offset_mirror_curvatures(self, group_idx, offsets)

    def set_focal_lengths(
        self,
        group_idx: int,
        focal_lengths: Array,
    ) -> Telescope:
        """Set mirror curvatures to achieve target focal lengths.

        For spherical/parabolic mirrors: c = 1/(2f).

        Args:
            group_idx: Index of mirror group
            focal_lengths: Target focal lengths array (N,)

        Returns:
            New Telescope with curvatures set for target focal lengths
        """
        from .operations import set_focal_lengths

        return set_focal_lengths(self, group_idx, focal_lengths)

    def get_mirrors_by_stage(self, stage: int) -> list[int]:
        """Get indices of mirror groups at a specific optical stage.

        Args:
            stage: Optical stage (0=primary, 1=secondary, etc.)

        Returns:
            List of mirror group indices at the specified stage
        """
        from .operations import get_mirrors_by_stage

        return get_mirrors_by_stage(self, stage)

    def get_mirror_count(self) -> int:
        """Get total number of mirrors across all groups.

        Returns:
            Total mirror count
        """
        from .operations import get_mirror_count

        return get_mirror_count(self)

    def add_sensor(self, sensor: SensorGroup) -> Telescope:
        """Add a new sensor group to the telescope.

        Args:
            sensor: SensorGroup to add

        Returns:
            New Telescope with added sensor group
        """
        from .operations import add_sensor

        return add_sensor(self, sensor)

    def replace_sensor(self, sensor: SensorGroup, idx: int = 0) -> Telescope:
        """Replace sensor group by index.

        Args:
            sensor: SensorGroup replacement
            idx: Index of sensor group to replace (default: 0)

        Returns:
            New Telescope with replaced sensor group

        Raises:
            IndexError: If index is out of range
        """
        from .operations import replace_sensor

        return replace_sensor(self, sensor, idx)

    def remove_sensor(self, idx: int = 0) -> Telescope:
        """Remove a sensor group by index.

        Args:
            idx: Index of sensor group to remove (default: 0)

        Returns:
            New Telescope with sensor group removed

        Raises:
            IndexError: If idx is out of range
        """
        from .operations import remove_sensor

        return remove_sensor(self, idx)

    def set_sensor_positions(self, idx: int, positions: Array) -> Telescope:
        """Set positions for all sensors in a group.

        Args:
            idx: Index of sensor group
            positions: New positions array (N, 3)

        Returns:
            New Telescope with updated sensor positions
        """
        from .operations import set_sensor_positions

        return set_sensor_positions(self, idx, positions)

    def set_sensor_rotations(self, idx: int, rotations: Array) -> Telescope:
        """Set rotations for all sensors in a group.

        Args:
            idx: Index of sensor group
            rotations: New rotations array (N, 3) Euler angles in degrees

        Returns:
            New Telescope with updated sensor rotations
        """
        from .operations import set_sensor_rotations

        return set_sensor_rotations(self, idx, rotations)

    def focus(self, delta_z: float, sensor_idx: int = 0) -> Telescope:
        """Adjust all sensor positions in a group along optical axis for focus.

        Args:
            delta_z: Distance to move sensors along z-axis (positive = away from mirrors)
            sensor_idx: Index of sensor group to adjust (default: 0)

        Returns:
            New Telescope with adjusted sensor positions
        """
        from .operations import focus

        return focus(self, delta_z, sensor_idx)

    def get_sensor_count(self) -> int:
        """Get number of sensor groups.

        Returns:
            Number of sensor groups
        """
        from .operations import get_sensor_count

        return get_sensor_count(self)

    def with_ste(self, sensor_idx: int = 0) -> Telescope:
        """Convert sensor group to straight-through estimator variant.

        Returns a new telescope with the specified sensor group converted to use
        straight-through estimation: hard assignment in forward pass,
        differentiable interpolation (bilinear for square, barycentric for hex)
        in backward pass.

        Args:
            sensor_idx: Index of sensor group to convert

        Returns:
            New Telescope with converted sensor group

        Raises:
            IndexError: If sensor_idx is out of range
            TypeError: If sensor type is not supported for conversion
        """
        from .operations import with_ste

        return with_ste(self, sensor_idx)

    def add_obstruction(self, obstruction: ObstructionGroup) -> Telescope:
        """Add an obstruction group to the telescope.

        Args:
            obstruction: Obstruction group to add

        Returns:
            New Telescope with obstruction group added
        """
        from .operations import add_obstruction

        return add_obstruction(self, obstruction)

    def remove_obstruction(self, group_idx: int) -> Telescope:
        """Remove an obstruction group by index.

        Args:
            group_idx: Index of obstruction group to remove

        Returns:
            New Telescope with obstruction group removed

        Raises:
            IndexError: If group_idx is out of range
        """
        from .operations import remove_obstruction

        return remove_obstruction(self, group_idx)

    def clear_obstructions(self) -> Telescope:
        """Remove all obstructions from telescope.

        Returns:
            New Telescope with no obstructions
        """
        from .operations import clear_obstructions

        return clear_obstructions(self)

    def get_obstruction_count(self) -> int:
        """Get total number of obstructions across all groups.

        Returns:
            Total obstruction count
        """
        from .operations import get_obstruction_count

        return get_obstruction_count(self)

    def clone(self) -> Telescope:
        """Create a deep copy of the telescope.

        Returns:
            Independent copy of the telescope
        """
        from .operations import clone

        return clone(self)

    def get_info(self) -> dict[str, Any]:
        """Get summary information about telescope configuration.

        Returns:
            Dictionary with telescope statistics and properties including:
            name, mirror counts, sensor info, obstruction count, bounding box.
        """
        from .operations import get_info

        return get_info(self)
