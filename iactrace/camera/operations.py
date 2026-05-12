from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from .camera import Camera
    from .concentrator import Concentrator
    from .photosensor import PhotoSensor


def _update_sensor_group(
    camera: Camera,
    sensor_idx: int,
    attr_getter: Callable,
    new_value: Any,
) -> Camera:
    """Replace a single attribute on a sensor group and return updated Camera."""
    new_sensor = eqx.tree_at(attr_getter, camera.sensor_groups[sensor_idx], new_value)
    new_groups = list(camera.sensor_groups)
    new_groups[sensor_idx] = new_sensor
    return eqx.tree_at(lambda c: c.sensor_groups, camera, new_groups)


def set_sensor_positions(
    camera: Camera, sensor_idx: int, positions: Any
) -> Camera:
    """Set positions for sensors in a group."""
    return _update_sensor_group(
        camera, sensor_idx, lambda s: s.positions, jnp.asarray(positions)
    )


def set_sensor_rotations(
    camera: Camera, sensor_idx: int, rotations: Any
) -> Camera:
    """Set rotations for sensors in a group."""
    return _update_sensor_group(
        camera, sensor_idx, lambda s: s.rotations, jnp.asarray(rotations)
    )


def set_concentrator(
    camera: Camera, concentrator: Concentrator | None
) -> Camera:
    """Set or replace the concentrator."""
    return eqx.tree_at(
        lambda c: c.concentrator, camera, concentrator,
        is_leaf=lambda x: x is None,
    )


def set_photosensor(camera: Camera, photosensor: PhotoSensor) -> Camera:
    """Set or replace the photosensor."""
    return eqx.tree_at(lambda c: c.photosensor, camera, photosensor)


def get_info(camera: Camera) -> dict[str, Any]:
    """Get summary information about camera configuration."""
    info: dict[str, Any] = {
        "photosensor_type": type(camera.photosensor).__name__,
        "n_sensor_groups": len(camera.sensor_groups),
        "has_concentrator": camera.concentrator is not None,
    }
    for i, sg in enumerate(camera.sensor_groups):
        info[f"sensor_group_{i}"] = {
            "type": type(sg).__name__,
            "n_sensors": sg.n_sensors,
            "accumulator_shape": sg.get_accumulator_shape(),
        }
    if camera.concentrator is not None:
        info["concentrator_type"] = type(camera.concentrator).__name__
    return info
