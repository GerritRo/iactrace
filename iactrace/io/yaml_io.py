from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml  # type: ignore[import-untyped]
from jax import Array
from pydantic import ValidationError

from ..camera import Camera
from ..telescope import Telescope
from .adapters import (
    camera_to_file_schema,
    lenses_from_schemas,
    mirrors_from_schemas,
    obstructions_from_schemas,
    sensor_from_schema,
    telescope_to_schema,
)
from .schemas import CameraFileSchema, TelescopeConfigSchema

logger = logging.getLogger(__name__)


class YAMLConfigError(Exception):
    """Raised when YAML configuration is invalid."""
    pass


# YAML helpers

def _make_precision_dumper(precision: int) -> type:
    """Return a ``yaml.SafeDumper`` subclass that formats floats to *precision*."""

    class PrecisionDumper(yaml.SafeDumper):
        pass

    def float_representer(dumper: PrecisionDumper, value: Any) -> yaml.ScalarNode:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:float", f"{value:.{precision}f}"
        )

    PrecisionDumper.add_representer(float, float_representer)
    PrecisionDumper.add_representer(np.float64, float_representer)
    PrecisionDumper.add_representer(np.float32, float_representer)
    return PrecisionDumper


def _write_yaml(config: dict[str, Any], filepath: Path, precision: int) -> None:
    """Dump *config* to *filepath* with controlled float precision."""
    dumper_cls = _make_precision_dumper(precision)
    with open(filepath, "w") as f:
        yaml.dump(config, f, Dumper=dumper_cls, default_flow_style=False, sort_keys=False)


def _check_overwrite(filepath: Path, overwrite: bool) -> None:
    if filepath.exists():
        if not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")
        logger.debug("Overwriting existing file: %s", filepath)


# Loading telescope


def load_telescope_config(
    filename: str | Path,
    n_samples: int = 100,
    *,
    key: Array,
) -> Telescope:
    """Load a telescope from a standalone telescope YAML file.

    The file describes optics (mirrors, lenses, obstructions) and the
    camera frame (``telescope.camera_position`` and
    ``telescope.camera_rotation``). A camera is loaded separately via
    :func:`load_camera_config` / :meth:`Camera.from_yaml`.

    Args:
        filename: Path to the telescope YAML file.
        n_samples: Number of Monte Carlo samples per mirror element.
        key: JAX random key for sampling and roughness.
    """
    with open(filename) as f:
        config = yaml.safe_load(f)

    return build_telescope_config(config, n_samples, key)


def build_telescope_config(
    config: dict[str, Any],
    n_samples: int,
    key: Array,
) -> Telescope:
    """Build a Telescope from a telescope configuration dictionary."""
    try:
        schema = TelescopeConfigSchema.model_validate(config)
    except ValidationError as e:
        raise YAMLConfigError(str(e)) from e

    try:
        key, mirror_key = jax.random.split(key)
        mirror_groups = mirrors_from_schemas(
            schema.mirrors, schema.mirror_templates, n_samples, key=mirror_key
        )
        obstruction_groups = obstructions_from_schemas(schema.obstructions)

        key, lens_key = jax.random.split(key)
        lens_groups = lenses_from_schemas(schema.lenses, key=lens_key)
    except ValueError as e:
        raise YAMLConfigError(str(e)) from e

    return Telescope(
        mirror_groups=mirror_groups,
        obstruction_groups=obstruction_groups,
        name=schema.telescope.name,
        lens_groups=lens_groups,
        camera_position=jnp.asarray(schema.telescope.camera_position),
        camera_rotation=jnp.asarray(schema.telescope.camera_rotation),
    )


# Loading camera


def load_camera_config(filename: str | Path) -> Camera:
    """Load a Camera from a standalone camera YAML file.

    Sensor positions in the file are interpreted as camera-local
    coordinates, so no telescope is needed.

    Args:
        filename: Path to camera YAML file.

    Returns:
        Camera object.
    """
    with open(filename) as f:
        raw = yaml.safe_load(f)

    return build_camera_config(raw)


def build_camera_config(config: dict[str, Any]) -> Camera:
    """Build a Camera from a camera configuration dictionary.

    Sensor positions are interpreted as camera-local coordinates.
    """
    try:
        schema = CameraFileSchema.model_validate(config)
    except ValidationError as e:
        raise YAMLConfigError(str(e)) from e

    sensors = [sensor_from_schema(s) for s in schema.sensors]

    return Camera(
        sensor_groups=sensors,
    )


# Serialization helpers


def telescope_to_dict(telescope: Telescope) -> dict[str, Any]:
    """Convert a Telescope to a telescope-only configuration dictionary."""
    schema = telescope_to_schema(telescope)
    return schema.model_dump(exclude_none=True)


def camera_to_dict(camera: Camera) -> dict[str, Any]:
    """Convert a Camera to a standalone configuration dictionary.

    Sensor positions are written in camera-local coordinates.

    Args:
        camera: The Camera object to convert.

    Returns:
        Configuration dictionary suitable for YAML serialization.
    """
    schema = camera_to_file_schema(camera)
    return schema.model_dump(exclude_none=True)


# Saving


def save_telescope(
    telescope: Telescope,
    filename: str | Path,
    precision: int = 8,
    overwrite: bool = True,
) -> Path:
    """Save a Telescope object to a standalone telescope YAML file.

    Args:
        telescope: The Telescope object to save.
        filename: Output file path.
        precision: Number of decimal places for float values.
        overwrite: If True, overwrite existing file.

    Returns:
        Path to the saved file.
    """
    filepath = Path(filename)
    _check_overwrite(filepath, overwrite)

    config = telescope_to_dict(telescope)
    _write_yaml(config, filepath, precision)

    logger.info("Saved telescope config to %s", filepath)
    return filepath


def save_camera(
    camera: Camera,
    filename: str | Path,
    precision: int = 6,
    overwrite: bool = True,
) -> Path:
    """Save a Camera to a standalone YAML file.

    Sensor positions are written in camera-local coordinates.

    Args:
        camera: The Camera object to save.
        filename: Output file path.
        precision: Number of decimal places for float values.
        overwrite: If True, overwrite existing file.

    Returns:
        Path to the saved file.
    """
    filepath = Path(filename)
    _check_overwrite(filepath, overwrite)

    config = camera_to_dict(camera)
    _write_yaml(config, filepath, precision)

    logger.info("Saved camera config to %s", filepath)
    return filepath
