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


# Reading files, with !include


_MERGE_KEY = "<<"
_MERGE_TAG = "tag:yaml.org,2002:merge"


class Include:
    """An unresolved ``!include path`` reference to another YAML file.

    load_yaml(..., resolve_includes=False) keeps these in place of the
    included content, and IncludeDumper writes them back as ``!include path``,
    so a config can be edited and saved without inlining its included files.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def __repr__(self) -> str:
        return f"Include({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Include) and other.path == self.path

    def __hash__(self) -> int:
        return hash(("Include", self.path))


def _apply_merges(data: Any) -> Any:
    """Fold every resolved ``<<: !include`` mapping into the mapping holding it."""
    if isinstance(data, list):
        return [_apply_merges(v) for v in data]
    if not isinstance(data, dict):
        return data
    out = {k: _apply_merges(v) for k, v in data.items()}
    base = out.get(_MERGE_KEY)
    if base is None or isinstance(base, Include):
        return out
    if not isinstance(base, dict):
        raise YAMLConfigError("'<<: !include' needs a file that holds a mapping")
    del out[_MERGE_KEY]
    return {**base, **out}


def load_yaml(
    filename: str | Path, resolve_includes: bool = True, _stack: tuple[Path, ...] = ()
) -> Any:
    """Load a YAML file that may pull in other files with ``!include``.

    ``key: !include other.yaml`` inserts the content of other.yaml as the
    value of key. ``<<: !include other.yaml`` merges the mapping in
    other.yaml into the mapping it appears in; keys written next to it
    override the included ones. Paths are relative to the file containing
    the tag, and included files may include further files.

    Parameters
    ----------
    filename
        Path to the YAML file.
    resolve_includes
        If False, leave Include placeholders instead of loading the files
        (a merge stays as an Include under the key ``<<``).
    """
    path = Path(filename).resolve()
    if path in _stack:
        raise YAMLConfigError(f"Circular !include of {path}")

    class Loader(yaml.SafeLoader):
        def flatten_mapping(self, node: yaml.MappingNode) -> None:
            # Keep "<<: !include file" as an ordinary entry; _apply_merges folds it in.
            for i, (key, value) in enumerate(node.value):
                if key.tag == _MERGE_TAG and value.tag == "!include":
                    node.value[i] = (yaml.ScalarNode("tag:yaml.org,2002:str", _MERGE_KEY), value)
            super().flatten_mapping(node)

    def include(loader: Loader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise YAMLConfigError(f"!include takes a file path ({path})")
        target = str(loader.construct_scalar(node))
        if not resolve_includes:
            return Include(target)
        included = path.parent / target
        if not included.is_file():
            raise YAMLConfigError(f"!include file not found: {included} (from {path})")
        return load_yaml(included, True, (*_stack, path))

    Loader.add_constructor("!include", include)
    with open(path) as f:
        data = yaml.load(f, Loader=Loader)  # noqa: S506 - a SafeLoader subclass
    return _apply_merges(data)


def resolve_includes(data: Any, base_dir: str | Path) -> Any:
    """Copy of data with every Include replaced by the file it points to.

    Paths are relative to base_dir, the directory of the file data came from.
    """

    def resolve(value: Any) -> Any:
        if isinstance(value, Include):
            return load_yaml(Path(base_dir) / value.path)
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return _apply_merges(resolve(data))


class IncludeDumper(yaml.SafeDumper):
    """SafeDumper that writes Include objects back as ``!include path``."""


def _represent_mapping(dumper: IncludeDumper, data: dict) -> yaml.MappingNode:
    node = yaml.SafeDumper.represent_dict(dumper, data)
    for key, value in node.value:
        if key.value == _MERGE_KEY and value.tag == "!include":
            key.tag = _MERGE_TAG  # written as a plain "<<" merge key
    return node


IncludeDumper.add_representer(
    Include, lambda dumper, value: dumper.represent_scalar("!include", value.path)
)
IncludeDumper.add_representer(dict, _represent_mapping)


# Dumping and file helpers


def _make_precision_dumper(precision: int) -> type:
    """Return a yaml.SafeDumper subclass that formats floats to precision."""

    class PrecisionDumper(IncludeDumper):
        pass

    def float_representer(dumper: PrecisionDumper, value: Any) -> yaml.ScalarNode:
        return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.{precision}f}")

    PrecisionDumper.add_representer(float, float_representer)
    PrecisionDumper.add_representer(np.float64, float_representer)
    PrecisionDumper.add_representer(np.float32, float_representer)
    return PrecisionDumper


def _write_yaml(config: dict[str, Any], filepath: Path, precision: int) -> None:
    """Dump config to filepath with controlled float precision."""
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
    camera frame (telescope.camera_position and
    telescope.camera_rotation). A camera is loaded separately via
    load_camera_config / Camera.from_yaml.

    Parameters
    ----------
    filename
        Path to the telescope YAML file.
    n_samples
        Number of Monte Carlo samples per mirror element.
    key
        JAX random key for sampling and roughness.
    """
    config = load_yaml(filename)

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

    Parameters
    ----------
    filename
        Path to camera YAML file.

    Returns
    -------
    Camera object.
    """
    raw = load_yaml(filename)

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

    return Camera(sensor_groups=sensors)


# Domain object -> dict


def telescope_to_dict(telescope: Telescope) -> dict[str, Any]:
    """Convert a Telescope to a telescope-only configuration dictionary."""
    schema = telescope_to_schema(telescope)
    return schema.model_dump(exclude_none=True)


def camera_to_dict(camera: Camera) -> dict[str, Any]:
    """Convert a Camera to a standalone configuration dictionary.

    Sensor positions are written in camera-local coordinates.

    Parameters
    ----------
    camera
        The Camera object to convert.

    Returns
    -------
    Configuration dictionary suitable for YAML serialization.
    """
    schema = camera_to_file_schema(camera)
    return schema.model_dump(exclude_none=True)


# Saving: dict -> file


def save_telescope(
    telescope: Telescope,
    filename: str | Path,
    precision: int = 8,
    overwrite: bool = True,
) -> Path:
    """Save a Telescope object to a standalone telescope YAML file.

    Parameters
    ----------
    telescope
        The Telescope object to save.
    filename
        Output file path.
    precision
        Number of decimal places for float values.
    overwrite
        If True, overwrite existing file.

    Returns
    -------
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

    Parameters
    ----------
    camera
        The Camera object to save.
    filename
        Output file path.
    precision
        Number of decimal places for float values.
    overwrite
        If True, overwrite existing file.

    Returns
    -------
    Path to the saved file.
    """
    filepath = Path(filename)
    _check_overwrite(filepath, overwrite)

    config = camera_to_dict(camera)
    _write_yaml(config, filepath, precision)

    logger.info("Saved camera config to %s", filepath)
    return filepath