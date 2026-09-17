"""

Configuration
=============

=========  ========================================================
Layer      Holds
=========  ========================================================
.yaml_io   open a file, validate, build, dump
.schemas   the on-disk grammar, as pydantic models
.adapters  schema <-> domain object, one module per thing described
=========  ========================================================
"""

from .aperture_table import (
    FORMAT,
    FORMAT_VERSION,
    load_aperture_table,
    read_aperture_table_arrays,
    save_aperture_table,
)
from .schemas import CameraFileSchema, TelescopeConfigSchema
from .yaml_io import (
    YAMLConfigError,
    build_camera_config,
    build_telescope_config,
    camera_to_dict,
    load_camera_config,
    load_telescope_config,
    save_camera,
    save_telescope,
    telescope_to_dict,
)

__all__ = [
    # -- Configuration: YAML in
    "load_telescope_config",
    "load_camera_config",
    "build_telescope_config",
    "build_camera_config",
    # -- Configuration: YAML out
    "save_telescope",
    "save_camera",
    "telescope_to_dict",
    "camera_to_dict",
    # -- Configuration: the schemas the files are validated against
    "TelescopeConfigSchema",
    "CameraFileSchema",
    "YAMLConfigError",
    # -- Results: effective-aperture tables
    "save_aperture_table",
    "load_aperture_table",
    "read_aperture_table_arrays",
    "FORMAT",
    "FORMAT_VERSION",
]
