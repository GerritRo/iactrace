"""The two top-level files: a telescope config and a camera config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..schemas import CameraFileSchema, TelescopeConfigSchema, TelescopeMetadataSchema
from ._common import _to_float_list
from .lenses import lenses_to_schemas
from .mirrors import mirrors_to_schemas
from .obstructions import obstructions_to_schemas
from .sensors import SensorSchemaType, sensors_to_schemas

if TYPE_CHECKING:
    from ...camera import Camera
    from ...telescope import Telescope


def telescope_to_schema(telescope: Telescope) -> TelescopeConfigSchema:
    """Convert a Telescope to a TelescopeConfigSchema (telescope-only file).

    The output describes mirrors/lenses/obstructions plus the camera frame
    (camera_position / camera_rotation). Detector geometry lives in
    a separate camera YAML; use camera_to_file_schema for that.
    """
    templates, mirror_schemas = mirrors_to_schemas(telescope.mirror_groups)

    cam_pos = np.asarray(telescope.camera_position)
    cam_rot = np.asarray(telescope.camera_rotation)

    return TelescopeConfigSchema(
        telescope=TelescopeMetadataSchema(
            name=telescope.name,
            camera_position=_to_float_list(cam_pos),
            camera_rotation=_to_float_list(cam_rot),
        ),
        mirror_templates=templates,
        mirrors=mirror_schemas,
        lenses=lenses_to_schemas(telescope.lens_groups),
        obstructions=obstructions_to_schemas(telescope.obstruction_groups),
    )


def camera_to_file_schema(camera: Camera) -> CameraFileSchema:
    """Convert a Camera to a standalone CameraFileSchema.

    Sensor positions are written in the camera-local frame.
    """
    sensor_schemas: list[SensorSchemaType] = []
    if camera.sensor_groups:
        sensor_schemas = sensors_to_schemas(camera.sensor_groups)

    return CameraFileSchema(sensors=sensor_schemas)
