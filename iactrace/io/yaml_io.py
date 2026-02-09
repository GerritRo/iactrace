from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml  # type: ignore[import-untyped]
from jax import Array

from ..telescope import Telescope

if TYPE_CHECKING:
    from ..core import Integrator
    from ..core.obstructions import ObstructionGroup
    from ..sensors import SensorGroup
    from ..telescope.lenses import LensGroup
    from ..telescope.mirrors import MirrorGroup


logger = logging.getLogger(__name__)


class YAMLConfigError(Exception):
    """Raised when YAML configuration is invalid."""
    pass


def load_telescope(
    filename: str | Path, integrator: Integrator, key: Array | None = None
) -> Telescope:
    """Load telescope from YAML configuration file.

    Args:
        filename: Path to YAML file.
        integrator: Integrator for sampling mirrors (e.g., MCIntegrator).
        key: JAX random key. Defaults to jax.random.key(0).

    Returns:
        Configured Telescope object.
    """
    if key is None:
        key = jax.random.key(0)

    with open(filename) as f:
        config = yaml.safe_load(f)

    return build_telescope(config, integrator, key)


def build_telescope(
    config: dict[str, Any], integrator: Integrator, key: Array
) -> Telescope:
    """Build telescope from configuration dictionary.

    Args:
        config: Configuration dictionary (from YAML or programmatic construction).
        integrator: Integrator for sampling mirrors.
        key: JAX random key.

    Returns:
        Configured Telescope object.
    """
    name = config.get("telescope", {}).get("name", "telescope")
    templates = config.get("mirror_templates", {})

    # Parse mirrors directly into groups
    mirror_groups = _parse_mirror_groups(config.get("mirrors", []), templates)

    # Only sample stage 0 (primary) mirrors
    # All groups get a unique perturbation_key for ray tracing roughness
    sampled_groups = []
    for group in mirror_groups:
        key, subkey = jax.random.split(key)
        if group.optical_stage == 0:
            sampled_groups.append(integrator.sample_group(group, subkey))
        else:
            # Stage 1+ mirrors don't need sampling, but set perturbation_key
            perturbation_key = jax.random.fold_in(subkey, 0xDEADBEEF)
            new_group = eqx.tree_at(lambda g: g.perturbation_key, group, perturbation_key)
            sampled_groups.append(new_group)

    obstruction_groups = _parse_obstruction_groups(config.get("obstructions", []))
    lens_groups = _parse_lens_groups(config.get("lenses", []))
    sensors = [_parse_sensor(s, i) for i, s in enumerate(config.get("sensors", []))]

    return Telescope(
        mirror_groups=sampled_groups,
        obstruction_groups=obstruction_groups,
        sensors=sensors,
        name=name,
        lens_groups=lens_groups if lens_groups else None,
    )


def save_telescope(
    telescope: Telescope,
    filename: str | Path,
    precision: int = 6,
    overwrite: bool = True,
) -> Path:
    """Save a Telescope object to a YAML file.

    Args:
        telescope: The Telescope object to save.
        filename: Output file path.
        precision: Number of decimal places for float values.
        overwrite: If True, overwrite existing file. If False, raise
                  FileExistsError if file exists.

    Returns:
        Path to the saved file.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    filepath = Path(filename)

    if filepath.exists():
        if not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")
        logger.debug("Overwriting existing file: %s", filepath)

    config = telescope_to_dict(telescope)

    # Create a custom dumper to avoid modifying global state
    class PrecisionDumper(yaml.SafeDumper):
        pass

    def float_representer(dumper: PrecisionDumper, value: Any) -> yaml.ScalarNode:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:float", f"{value:.{precision}f}"
        )

    PrecisionDumper.add_representer(float, float_representer)
    PrecisionDumper.add_representer(np.float64, float_representer)
    PrecisionDumper.add_representer(np.float32, float_representer)

    with open(filepath, "w") as f:
        yaml.dump(
            config, f, Dumper=PrecisionDumper, default_flow_style=False, sort_keys=False
        )

    logger.info("Saved telescope config to %s", filepath)
    return filepath


def telescope_to_dict(telescope: Telescope) -> dict[str, Any]:
    """Convert a Telescope object to a configuration dictionary.

    Args:
        telescope: The Telescope object to convert.

    Returns:
        Configuration dictionary suitable for YAML serialization.
    """
    config: dict[str, Any] = {
        "telescope": {"name": telescope.name, "units": "m"},
        "mirror_templates": {},
        "mirrors": [],
        "lenses": [],
        "obstructions": [],
        "sensors": [],
    }

    # Extract mirror templates and mirrors
    templates, mirrors = _extract_mirrors(telescope)
    config["mirror_templates"] = templates
    config["mirrors"] = mirrors

    # Extract lenses
    if telescope.lens_groups:
        config["lenses"] = _extract_lenses(telescope)

    # Extract obstructions
    if telescope.obstruction_groups:
        config["obstructions"] = _extract_obstructions(telescope)

    # Extract sensors
    if telescope.sensors:
        config["sensors"] = _extract_sensors(telescope)

    return config


def _ensure_ccw(vertices: Array) -> Array:
    """Ensure polygon vertices are in counter-clockwise order."""
    vx, vy = vertices[:, 0], vertices[:, 1]
    signed_area = 0.5 * jnp.sum(vx * jnp.roll(vy, -1) - jnp.roll(vx, -1) * vy)
    return jnp.where(signed_area < 0, vertices[::-1], vertices)


def _parse_mirror_groups(mirrors_config: list[dict], templates: dict) -> list[MirrorGroup]:
    """Parse mirror configs into MirrorGroup objects."""
    from .serialization import mirror_registry

    if not mirrors_config:
        return []

    # Parse each mirror's data
    parsed = []
    for i, m in enumerate(mirrors_config):
        mirror_id = m.get("id", f"mirror[{i}]")

        # Validate required fields
        if "aperture" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' is missing required 'aperture' field"
            )
        if "template" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' is missing required 'template' field"
            )

        template_name = m["template"]
        if template_name not in templates:
            available = ", ".join(templates.keys()) if templates else "(none defined)"
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' references undefined template '{template_name}'. "
                f"Available templates: {available}"
            )

        aperture_config = m["aperture"]
        template = templates[template_name]

        if "surface" not in template:
            raise YAMLConfigError(
                f"Template '{template_name}' is missing required 'surface' field"
            )
        template_surface = template["surface"]

        # Validate aperture type
        if "type" not in aperture_config:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' aperture is missing required 'type' field"
            )
        aperture_type = aperture_config["type"]
        if aperture_type not in ("circular", "polygon"):
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' has unknown aperture type '{aperture_type}'. "
                f"Expected 'circular' or 'polygon'"
            )

        # Validate aperture-specific fields
        if aperture_type == "circular" and "radius" not in aperture_config:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' circular aperture is missing required 'radius' field"
            )
        if aperture_type == "polygon" and "vertices" not in aperture_config:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' polygon aperture is missing required 'vertices' field"
            )

        # Validate required position/orientation
        if "position" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' is missing required 'position' field"
            )
        if "orientation" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}' is missing required 'orientation' field"
            )

        # Validate template surface fields
        if "curvature" not in template_surface and "curvature" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}': neither template '{template_name}' nor mirror "
                f"definition specifies 'curvature'"
            )
        if "conic" not in template_surface and "conic" not in m:
            raise YAMLConfigError(
                f"Mirror '{mirror_id}': neither template '{template_name}' nor mirror "
                f"definition specifies 'conic'"
            )

        # Per-mirror surface parameters (can override template)
        curvature = float(m.get("curvature", template_surface["curvature"]))
        conic = float(m.get("conic", template_surface["conic"]))
        aspheric = m.get("aspheric", template_surface.get("aspheric", []))

        parsed.append({
            "position": jnp.asarray(m["position"]),
            "rotation": jnp.asarray(m["orientation"]),
            "curvature": curvature,
            "conic": conic,
            "aspheric": jnp.array(aspheric, dtype=jnp.float32),
            "offset": jnp.asarray(m.get("offset", [0.0, 0.0])),
            "stage": m.get("stage", 0),
            "aperture_type": aperture_config["type"],
            "aperture_data": aperture_config,
        })

    groups: list[MirrorGroup] = []

    # Group by stage
    by_stage = defaultdict(list)
    for p in parsed:
        by_stage[p["stage"]].append(p)

    for stage, stage_mirrors in sorted(by_stage.items()):
        # Separate by aperture type
        disk_mirrors = [m for m in stage_mirrors if m["aperture_type"] == "circular"]
        poly_mirrors = [m for m in stage_mirrors if m["aperture_type"] == "polygon"]

        if disk_mirrors:
            groups.append(_build_disk_group(disk_mirrors, stage, mirror_registry))

        # Group polygon mirrors by vertex count
        if poly_mirrors:
            by_nverts = defaultdict(list)
            for m in poly_mirrors:
                n_verts = len(m["aperture_data"]["vertices"])
                by_nverts[n_verts].append(m)

            for _n_verts, mirror_list in by_nverts.items():
                groups.append(_build_polygon_group(mirror_list, stage, mirror_registry))

    return groups


def _build_disk_group(mirrors: list[dict], stage: int, registry) -> MirrorGroup:
    """Build circular aperture MirrorGroup."""
    mirror_class = registry.get("circular")

    configs = []
    for m in mirrors:
        config = {
            "position": m["position"].tolist(),
            "orientation": m["rotation"].tolist(),
            "aperture": m["aperture_data"],
            "offset": m["offset"].tolist(),
            "curvature": m["curvature"],
            "conic": m["conic"],
            "aspheric": m["aspheric"].tolist(),
        }
        configs.append(config)

    return mirror_class.from_config(configs, {}, optical_stage=stage)


def _build_polygon_group(mirrors: list[dict], stage: int, registry) -> MirrorGroup:
    """Build polygon aperture MirrorGroup."""
    mirror_class = registry.get("polygon")

    configs = []
    for m in mirrors:
        vertices = _ensure_ccw(jnp.asarray(m["aperture_data"]["vertices"]))
        config = {
            "position": m["position"].tolist(),
            "orientation": m["rotation"].tolist(),
            "aperture": {
                "type": "polygon",
                "vertices": vertices.tolist(),
            },
            "offset": m["offset"].tolist(),
            "curvature": m["curvature"],
            "conic": m["conic"],
            "aspheric": m["aspheric"].tolist(),
        }
        configs.append(config)

    return mirror_class.from_config(configs, {}, optical_stage=stage)


def _parse_obstruction_groups(obstructions_config: list[dict]) -> list[ObstructionGroup]:
    """Parse obstruction configs into ObstructionGroup objects."""
    from .serialization import obstruction_registry

    if not obstructions_config:
        return []

    by_type: dict[str, list[dict]] = defaultdict(list)
    for o in obstructions_config:
        otype = o["type"]
        by_type[otype].append(o)

    groups: list[ObstructionGroup] = []
    for otype, configs in by_type.items():
        obs_class = obstruction_registry.get(otype)
        groups.append(obs_class.from_config(configs))

    return groups


def _parse_lens_groups(lenses_config: list[dict]) -> list[LensGroup]:
    """Parse lens configs into LensGroup objects."""
    from .serialization import lens_registry

    if not lenses_config:
        return []

    by_type: dict[str, list[dict]] = defaultdict(list)
    for lens in lenses_config:
        ltype = lens["type"]
        by_type[ltype].append(lens)

    groups: list[LensGroup] = []
    for ltype, configs in by_type.items():
        lens_class = lens_registry.get(ltype)
        # Extract optical_stage from first config if present
        optical_stage = configs[0].get("optical_stage", 0) if configs else 0
        groups.append(lens_class.from_config(configs, optical_stage=optical_stage))

    return groups


def _parse_sensor(config: dict[str, Any], index: int = 0) -> SensorGroup:
    """Parse sensor config into a SensorGroup object.

    Each sensor config creates a single-sensor group. Multiple sensors in the
    same group (e.g., CTAO SST camera with 64 sensors) should be defined
    programmatically, not via YAML.
    """
    from .serialization import sensor_registry

    sensor_id = config.get("id", f"sensor[{index}]")

    if "type" not in config:
        raise YAMLConfigError(
            f"Sensor '{sensor_id}' is missing required 'type' field"
        )

    stype = config["type"]

    # Validate common required fields
    if "position" not in config:
        raise YAMLConfigError(
            f"Sensor '{sensor_id}' is missing required 'position' field"
        )
    if "orientation" not in config:
        raise YAMLConfigError(
            f"Sensor '{sensor_id}' is missing required 'orientation' field"
        )

    # Validate type-specific fields
    if stype == "square":
        for field in ("width", "height", "bounds"):
            if field not in config:
                raise YAMLConfigError(
                    f"Square sensor '{sensor_id}' is missing required '{field}' field"
                )
        if len(config["bounds"]) != 4:
            raise YAMLConfigError(
                f"Square sensor '{sensor_id}' bounds must have 4 values "
                f"(x_min, x_max, y_min, y_max), got {len(config['bounds'])}"
            )
    elif stype == "hexagonal":
        if "centers_x" not in config or "centers_y" not in config:
            raise YAMLConfigError(
                f"Hexagonal sensor '{sensor_id}' is missing required "
                f"'centers_x' and/or 'centers_y' fields"
            )
        if len(config["centers_x"]) != len(config["centers_y"]):
            raise YAMLConfigError(
                f"Hexagonal sensor '{sensor_id}': centers_x and centers_y "
                f"must have same length"
            )

    # Get sensor class from registry
    try:
        sensor_class = sensor_registry.get(stype)
    except KeyError:
        available = sensor_registry.registered_types()
        raise YAMLConfigError(
            f"Sensor '{sensor_id}' has unknown type '{stype}'. "
            f"Available types: {', '.join(available)}"
        ) from None

    return sensor_class.from_config([config])


def _extract_mirrors(telescope: Telescope) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract mirror templates and mirror configurations from telescope."""
    templates: dict[str, Any] = {}
    mirrors: list[dict[str, Any]] = []
    template_counter = 0
    surface_to_template: dict[tuple, str] = {}

    for group in telescope.mirror_groups:
        n_mirrors = len(group)

        for i in range(n_mirrors):
            surface_params = group.get_surface_params(i)
            curvature = surface_params["curvature"]
            conic = surface_params["conic"]
            aspheric = surface_params["aspheric"]

            surface_key = (curvature, conic, tuple(aspheric))

            if surface_key not in surface_to_template:
                template_name = f"template_{template_counter}"
                template_counter += 1
                surface_to_template[surface_key] = template_name
                templates[template_name] = {
                    "surface": {
                        "curvature": curvature,
                        "conic": conic,
                        "aspheric": aspheric if aspheric else [],
                    }
                }

            template_name = surface_to_template[surface_key]

            mirror_config = group.to_config(i)
            mirror_config["id"] = f"M_{len(mirrors)}"
            mirror_config["template"] = template_name

            mirrors.append(mirror_config)

    return templates, mirrors


def _extract_obstructions(telescope: Telescope) -> list[dict[str, Any]]:
    """Extract obstruction configurations from telescope."""
    obstructions: list[dict[str, Any]] = []
    obs_counter = 0

    if not telescope.obstruction_groups:
        return obstructions

    for group in telescope.obstruction_groups:
        n_obs = len(group)

        for i in range(n_obs):
            obs_config = group.to_config(i)
            obs_config["id"] = f"obs_{obs_counter}"
            obs_counter += 1
            obstructions.append(obs_config)

    return obstructions


def _extract_lenses(telescope: Telescope) -> list[dict[str, Any]]:
    """Extract lens configurations from telescope."""
    lenses: list[dict[str, Any]] = []
    lens_counter = 0

    if not telescope.lens_groups:
        return lenses

    for group in telescope.lens_groups:
        n_lenses = len(group)

        for i in range(n_lenses):
            lens_config = group.to_config(i)
            lens_config["id"] = f"lens_{lens_counter}"
            lens_config["optical_stage"] = group.optical_stage
            lens_counter += 1
            lenses.append(lens_config)

    return lenses


def _extract_sensors(telescope: Telescope) -> list[dict[str, Any]]:
    """Extract sensor configurations from telescope.

    Each sensor in each sensor group is extracted as a separate config dict.
    """
    sensors: list[dict[str, Any]] = []
    sensor_counter = 0

    for group in telescope.sensors:
        n_sensors = len(group)
        for i in range(n_sensors):
            sensor_config = group.to_config(i)
            sensor_config["id"] = f"sensor_{sensor_counter}"
            sensor_counter += 1
            sensors.append(sensor_config)

    return sensors
