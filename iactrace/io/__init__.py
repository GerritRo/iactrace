"""I/O module for loading and saving telescope configurations."""

from .yaml_io import (
    YAMLConfigError,
    build_telescope,
    load_telescope,
    save_telescope,
    telescope_to_dict,
)

__all__ = [
    "load_telescope",
    "build_telescope",
    "save_telescope",
    "telescope_to_dict",
    "YAMLConfigError",
]
