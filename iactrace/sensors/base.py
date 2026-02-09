from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
from jax import Array

if TYPE_CHECKING:
    from typing import Self


class SensorGroup(eqx.Module):
    """Abstract base class for sensor groups.

    A sensor group contains N sensors at different positions/orientations that share
    the same pixel geometry. During ray tracing, rays are intersected with all sensor
    planes and the closest hit is kept. This enables modeling complex sensor arrangements
    like the CTAO SST with 64 planar sensors distributed over a curved surface.

    All sensors in a group must have:
    - positions: 3D positions in world coordinates (N, 3)
    - rotations: Euler angles (tip, tilt, rotation) in degrees (N, 3)
    - get_accumulator_shape(): Returns shape of output per sensor
    - accumulate(sensor_idx, x, y, values): Accumulates values into pixels

    The output shape is always (n_sensors, *per_sensor_shape), e.g.:
    - For square sensors: (N, height, width)
    - For hexagonal sensors: (N, n_pixels)

    Attributes:
        positions: Sensor positions in 3D space (N, 3)
        rotations: Sensor rotations as Euler angles in degrees (N, 3)
        config_type: Type identifier used in YAML configs
    """

    positions: Array
    rotations: Array
    config_type: ClassVar[str]

    @property
    def n_sensors(self) -> int:
        """Return number of sensors in the group."""
        return self.positions.shape[0]

    def __len__(self) -> int:
        """Return number of sensors in the group."""
        return self.n_sensors

    @abstractmethod
    def get_accumulator_shape(self) -> tuple[int, ...]:
        """Return the shape of the accumulator array per sensor.

        Returns:
            Shape tuple for the output image/accumulator per sensor.
            For square sensors: (height, width)
            For hexagonal sensors: (n_pixels,)

        Note:
            The full output shape is (n_sensors, *get_accumulator_shape()).
        """
        raise NotImplementedError

    @abstractmethod
    def accumulate(
        self, sensor_idx: Array, x: Array, y: Array, values: Array
    ) -> Array:
        """Accumulate values at given positions into pixels across all sensors.

        Args:
            sensor_idx: Index of sensor each ray hit (n_rays,)
            x: X coordinates of hit positions in local sensor planes (n_rays,)
            y: Y coordinates of hit positions in local sensor planes (n_rays,)
            values: Values to accumulate (e.g., photon weights) (n_rays,)

        Returns:
            Accumulated images with shape (n_sensors, *get_accumulator_shape())
        """
        raise NotImplementedError

    @abstractmethod
    def to_config(self, index: int = 0) -> dict[str, Any]:
        """Convert a single sensor at index to a configuration dictionary.

        Args:
            index: Index of sensor to convert (default: 0)

        Returns:
            Configuration dictionary for YAML serialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> Self:
        """Create a sensor group from a list of configuration dictionaries.

        All sensors in the group must share the same pixel geometry (width, height,
        bounds for square sensors; hex_centers for hexagonal sensors).

        Args:
            configs: List of configuration dictionaries from YAML.

        Returns:
            New sensor group instance containing all sensors.
        """
        raise NotImplementedError
