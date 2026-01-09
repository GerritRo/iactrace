from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..core import render, render_debug

if TYPE_CHECKING:
    from pathlib import Path

    from ..core import Integrator, ObstructionGroup
    from ..sensors import HexagonalSensor, SquareSensor

    from .mirrors import MirrorGroup

    Sensor = Union[SquareSensor, HexagonalSensor]


class Telescope(eqx.Module):
    """
    IACT telescope configuration as an Equinox Module.

    Stores mirrors, obstructions, and sensors as object lists for
    polymorphic dispatch while maintaining JAX compatibility.
    """

    mirror_groups: list[MirrorGroup]
    obstruction_groups: list[ObstructionGroup] | None
    sensors: list[Sensor]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        mirror_groups: list[MirrorGroup],
        obstruction_groups: list[ObstructionGroup] | None = None,
        sensors: list[Sensor] | None = None,
        name: str = "telescope",
    ) -> None:
        """
        Initialize Telescope.

        Args:
            mirror_groups: List of Mirror groups
            obstruction_groups: List of Obstruction groups
            sensors: List of sensor objects
            name: Telescope name
        """
        self.mirror_groups = mirror_groups
        self.obstruction_groups = obstruction_groups
        self.sensors = list(sensors) if sensors else []
        self.name = name

    def __call__(
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

    @classmethod
    def from_yaml(
        cls,
        filename: str | Path,
        integrator: Integrator,
        key: Array | None = None,
    ) -> Telescope:
        """Load from YAML config."""
        from ..io.yaml_loader import load_telescope

        return load_telescope(filename, integrator, key)

    # Convenience methods

    def resample_mirrors(self, integrator: Integrator, key: Array) -> Telescope:
        """Resample all mirrors with specified integrator."""
        from .operations import resample_mirrors

        return resample_mirrors(self, integrator, key)

    def set_mirror_positions(self, group_idx: int, positions: Array) -> Telescope:
        """Set positions for all mirrors in a group."""
        from .operations import set_mirror_positions

        return set_mirror_positions(self, group_idx, positions)

    def set_mirror_rotations(self, group_idx: int, rotations: Array) -> Telescope:
        """Set rotations for all mirrors in a group."""
        from .operations import set_mirror_rotations

        return set_mirror_rotations(self, group_idx, rotations)

    def scale_mirror_weights(
        self, group_idx: int, scale_factors: Array | float
    ) -> Telescope:
        """Scale reflectivity weights for mirrors in a group."""
        from .operations import scale_mirror_weights

        return scale_mirror_weights(self, group_idx, scale_factors)

    def apply_roughness(self, roughness_arcsec: float) -> Telescope:
        """Apply mirror roughness in arcsec to all mirrors."""
        from .operations import apply_roughness

        return apply_roughness(self, roughness_arcsec)

    def apply_roughness_to_group(self, group_idx: int, roughness: float) -> Telescope:
        """Apply roughness to a specific mirror group."""
        from .operations import apply_roughness_to_group

        return apply_roughness_to_group(self, group_idx, roughness)

    def get_mirrors_by_stage(self, stage: int) -> list[int]:
        """Get indices of mirror groups at a specific optical stage."""
        from .operations import get_mirrors_by_stage

        return get_mirrors_by_stage(self, stage)

    def get_mirror_count(self) -> int:
        """Get total number of mirrors across all groups."""
        from .operations import get_mirror_count

        return get_mirror_count(self)

    def add_sensor(self, sensor: Sensor) -> Telescope:
        """Add a new sensor to the telescope."""
        from .operations import add_sensor

        return add_sensor(self, sensor)

    def replace_sensor(self, sensor: Sensor, idx: int = 0) -> Telescope:
        """Replace a sensor by index."""
        from .operations import replace_sensor

        return replace_sensor(self, sensor, idx)

    def remove_sensor(self, idx: int = 0) -> Telescope:
        """Remove a sensor by index."""
        from .operations import remove_sensor

        return remove_sensor(self, idx)

    def set_sensor_position(self, idx: int, position: Array) -> Telescope:
        """Set position of a sensor."""
        from .operations import set_sensor_position

        return set_sensor_position(self, idx, position)

    def set_sensor_rotation(self, idx: int, rotation: Array) -> Telescope:
        """Set rotation of a sensor."""
        from .operations import set_sensor_rotation

        return set_sensor_rotation(self, idx, rotation)

    def focus(self, delta_z: float, sensor_idx: int = 0) -> Telescope:
        """Adjust sensor position along optical axis for focus."""
        from .operations import focus

        return focus(self, delta_z, sensor_idx)

    def get_sensor_count(self) -> int:
        """Get number of sensors."""
        from .operations import get_sensor_count

        return get_sensor_count(self)

    def add_obstruction(self, obstruction: ObstructionGroup) -> Telescope:
        """Add an obstruction group to the telescope."""
        from .operations import add_obstruction

        return add_obstruction(self, obstruction)

    def remove_obstruction(self, group_idx: int) -> Telescope:
        """Remove an obstruction group by index."""
        from .operations import remove_obstruction

        return remove_obstruction(self, group_idx)

    def clear_obstructions(self) -> Telescope:
        """Remove all obstructions from telescope."""
        from .operations import clear_obstructions

        return clear_obstructions(self)

    def get_obstruction_count(self) -> int:
        """Get number of obstructions."""
        from .operations import get_obstruction_count

        return get_obstruction_count(self)

    def clone(self) -> Telescope:
        """Create a deep copy of the telescope."""
        from .operations import clone

        return clone(self)

    def get_info(self) -> dict[str, Any]:
        """Get summary information about telescope configuration."""
        from .operations import get_info

        return get_info(self)