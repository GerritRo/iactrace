from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp

from .chain import DetectionChain

if TYPE_CHECKING:
    from .camera import Camera
    from .concentrator import Concentrator
    from .photosensor import PhotoSensor
    from .sensor_group import SensorGroup


def _update_sensor_group(
    camera: Camera,
    sensor_idx: int,
    attr_getter: Callable,
    new_value: Any,
) -> Camera:
    """Replace a single attribute on a sensor group and return updated Camera."""
    new_sensor = eqx.tree_at(attr_getter, camera.sensor_groups[sensor_idx], new_value)
    return _replace_sensor_group(camera, sensor_idx, new_sensor)


def _replace_sensor_group(camera: Camera, sensor_idx: int, new_group: SensorGroup) -> Camera:
    """Swap a whole sensor group into the camera and return the updated Camera."""
    new_groups = list(camera.sensor_groups)
    new_groups[sensor_idx] = new_group
    return eqx.tree_at(lambda c: c.sensor_groups, camera, new_groups)


def set_sensor_positions(camera: Camera, sensor_idx: int, positions: Any) -> Camera:
    """Set positions for sensors in a group."""
    return _update_sensor_group(camera, sensor_idx, lambda s: s.positions, jnp.asarray(positions))


def set_sensor_rotations(camera: Camera, sensor_idx: int, rotations: Any) -> Camera:
    """Set rotations for sensors in a group."""
    return _update_sensor_group(camera, sensor_idx, lambda s: s.rotations, jnp.asarray(rotations))


def _replace_chain(camera: Camera, sensor_idx: int, new_chain: DetectionChain) -> Camera:
    """Swap a sensor group's detection chain and return the updated Camera."""
    group = camera.sensor_groups[sensor_idx]
    new_group = eqx.tree_at(lambda g: g.chain, group, new_chain)
    return _replace_sensor_group(camera, sensor_idx, new_group)


def set_concentrator(camera: Camera, sensor_idx: int, concentrator: Concentrator | None) -> Camera:
    """Set or replace the concentrator on a sensor group's detection chain."""
    chain = camera.sensor_groups[sensor_idx].chain
    return _replace_chain(
        camera,
        sensor_idx,
        DetectionChain(concentrator, chain.photosensor, chain.gap),
    )


def set_photosensor(camera: Camera, sensor_idx: int, photosensor: PhotoSensor) -> Camera:
    """Set or replace the photosensor on a sensor group's detection chain."""
    chain = camera.sensor_groups[sensor_idx].chain
    return _replace_chain(
        camera,
        sensor_idx,
        DetectionChain(chain.concentrator, photosensor, chain.gap),
    )


def set_gap(camera: Camera, sensor_idx: int, gap: float) -> Camera:
    """Set the gap (upstream exit -> detector spacing) on a group's chain."""
    chain = camera.sensor_groups[sensor_idx].chain
    return _replace_chain(
        camera,
        sensor_idx,
        DetectionChain(chain.concentrator, chain.photosensor, float(gap)),
    )


def get_info(camera: Camera) -> dict[str, Any]:
    """Get summary information about camera configuration.

    Each sensor group reports its own geometry and detection chain, since the
    chain (concentrator + gap + photosensor) is owned per group.
    """
    info: dict[str, Any] = {
        "n_sensor_groups": len(camera.sensor_groups),
    }
    for i, sg in enumerate(camera.sensor_groups):
        chain = sg.chain
        group_info: dict[str, Any] = {
            "type": type(sg).__name__,
            "n_sensors": sg.n_sensors,
            "accumulator_shape": sg.get_accumulator_shape(),
            "photosensor_type": type(chain.photosensor).__name__,
            "has_concentrator": chain.concentrator is not None,
            "gap": float(chain.gap),
            "detector_z": float(chain.detector_z),
        }
        if chain.concentrator is not None:
            group_info["concentrator_type"] = type(chain.concentrator).__name__
        info[f"sensor_group_{i}"] = group_info
    return info
