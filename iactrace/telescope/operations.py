from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from ..core import Integrator, ObstructionGroup
    from ..sensors import SensorGroup
    from .telescope import Telescope


def _update_mirror_group_attr(
    telescope: Telescope,
    group_idx: int,
    attr_getter: Callable,
    new_value: Any,
) -> Telescope:
    """Update a single attribute on a mirror group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        attr_getter: Lambda to get the attribute (e.g., lambda g: g.positions)
        new_value: New value for the attribute

    Returns:
        New Telescope with updated mirror group
    """
    new_group = eqx.tree_at(attr_getter, telescope.mirror_groups[group_idx], new_value)
    new_groups = list(telescope.mirror_groups)
    new_groups[group_idx] = new_group
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def _update_sensor_attr(
    telescope: Telescope,
    sensor_idx: int,
    attr_getter: Callable,
    new_value: Any,
) -> Telescope:
    """Update a single attribute on a sensor group.

    Args:
        telescope: Telescope instance
        sensor_idx: Index of sensor group to modify
        attr_getter: Lambda to get the attribute (e.g., lambda s: s.positions)
        new_value: New value for the attribute

    Returns:
        New Telescope with updated sensor group
    """
    new_sensor = eqx.tree_at(attr_getter, telescope.sensors[sensor_idx], new_value)
    new_sensors = list(telescope.sensors)
    new_sensors[sensor_idx] = new_sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def resample_mirrors(telescope: Telescope, integrator: Integrator, key: Array) -> Telescope:
    """Resample all mirrors with specified integrator

    Args:
        telescope: Telescope instance
        integrator: Integrator object
        key: jax.random.key

    Returns:
        New telescope with resampled mirrors
    """
    keys = jax.random.split(key, len(telescope.mirror_groups))
    new_groups = [
        integrator.sample_group(g, k) for g, k in zip(telescope.mirror_groups, keys, strict=False)
    ]
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def set_mirror_positions(
    telescope: Telescope, group_idx: int, positions: Array
) -> Telescope:
    """Set positions for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        positions: New positions array (N, 3)

    Returns:
        New Telescope with updated mirror positions
    """
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.positions, jnp.asarray(positions)
    )


def set_mirror_rotations(
    telescope: Telescope, group_idx: int, rotations: Array
) -> Telescope:
    """Set rotations for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        rotations: New rotations array (N, 3) in degrees (Euler angles)

    Returns:
        New Telescope with updated mirror rotations
    """
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.rotations, jnp.asarray(rotations)
    )


def scale_mirror_weights(
    telescope: Telescope, group_idx: int, scale_factors: Array | float
) -> Telescope:
    """Scale reflectivity for mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        scale_factors: Scale factors per mirror (N,) or single value

    Returns:
        New Telescope with scaled mirror reflectivity
    """
    scale_factors = jnp.asarray(scale_factors)
    if scale_factors.ndim == 0:
        scale_factors = jnp.full(
            len(telescope.mirror_groups[group_idx]), scale_factors
        )

    current_reflectivity = telescope.mirror_groups[group_idx].reflectivity
    new_reflectivity = current_reflectivity * scale_factors

    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.reflectivity, new_reflectivity
    )


def apply_roughness(telescope: Telescope, roughness: float) -> Telescope:
    """Apply roughness to all telescope mirrors

    Args:
        telescope: Telescope instance
        roughness: Surface roughness in arcseconds

    Returns:
        New Telescope with updated roughness for mirrors
    """
    sigma_rad = roughness * jnp.pi / (180.0 * 3600.0)
    new_groups = []
    for group in telescope.mirror_groups:
        new_scale = jnp.full(len(group), sigma_rad)
        new_groups.append(
            eqx.tree_at(lambda g: g.perturbation_scale, group, new_scale)
        )
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def apply_roughness_to_group(
    telescope: Telescope, group_idx: int, roughness: float
) -> Telescope:
    """Apply roughness to a specific mirror group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        roughness: Surface roughness in arcseconds

    Returns:
        New Telescope with updated roughness for specified group
    """
    sigma_rad = roughness * jnp.pi / (180.0 * 3600.0)
    new_scale = jnp.full(len(telescope.mirror_groups[group_idx]), sigma_rad)
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.perturbation_scale, new_scale
    )


def apply_misalignment_to_group(telescope, group_idx: int, sigma_h: float, sigma_v: float, key: Array) -> Telescope:
    """Apply random Gaussian misalignment to mirror orientations.

    Adds random perturbations to the horizontal and vertical
    angles of each mirror in the specified group, drawn from independent
    Gaussian distributions.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        sigma_h: Standard deviation of horizontal misalignment in arcseconds
        sigma_v: Standard deviation of vertical misalignment in arcseconds
        key: JAX random key for reproducibility

    Returns:
        New Telescope with randomly misaligned mirrors
    """
    group = telescope.mirror_groups[group_idx]
    n_mirrors = len(group)

    # Convert arcseconds to degrees (rotations are stored in degrees)
    sigma_h_deg = sigma_h / 3600.0
    sigma_v_deg = sigma_v / 3600.0

    # Generate random misalignments
    key1, key2 = jax.random.split(key)
    delta_h = jax.random.normal(key1, shape=(n_mirrors,)) * sigma_h_deg
    delta_v = jax.random.normal(key2, shape=(n_mirrors,)) * sigma_v_deg

    # Apply to rotations:
    current_rotations = group.rotations
    new_rotations = current_rotations.at[:, 0].add(delta_v)
    new_rotations = new_rotations.at[:, 1].add(delta_h)

    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.rotations, new_rotations
    )


def apply_displacement_to_group(telescope, group_idx: int, sigma_z: float, key: Array) -> Telescope:
    """Apply random Gaussian distance adjustment to mirrors along the z-axis.

    Adds random perturbations to the z-coordinate of each mirror position
    in the specified group, drawn from a Gaussian distribution.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        sigma_z: Standard deviation of z-axis displacement (same units as positions)
        key: JAX random key for reproducibility

    Returns:
        New Telescope with randomly displaced mirrors
    """
    group = telescope.mirror_groups[group_idx]
    n_mirrors = len(group)

    # Generate random z displacements
    delta_z = jax.random.normal(key, shape=(n_mirrors,)) * sigma_z

    # Apply to z-component of positions
    current_positions = group.positions
    new_positions = current_positions.at[:, 2].add(delta_z)

    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.positions, new_positions
    )


def set_mirror_curvatures(
    telescope,
    group_idx: int,
    curvatures: Array,
) -> Telescope:
    """Set curvatures for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        curvatures: New curvatures array (N,)

    Returns:
        New Telescope with updated mirror curvatures
    """
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.curvatures, jnp.asarray(curvatures)
    )


def set_mirror_conics(
    telescope,
    group_idx: int,
    conics: Array,
) -> Telescope:
    """Set conic constants for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        conics: New conic constants array (N,)

    Returns:
        New Telescope with updated mirror conic constants
    """
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.conics, jnp.asarray(conics)
    )


def set_mirror_aspherics(
    telescope,
    group_idx: int,
    aspherics: Array,
) -> Telescope:
    """Set aspheric coefficients for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        aspherics: New aspheric coefficients array (N, K) where K is number of terms

    Returns:
        New Telescope with updated mirror aspheric coefficients
    """
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.aspherics, jnp.asarray(aspherics)
    )


def scale_mirror_curvatures(
    telescope,
    group_idx: int,
    scale_factors: Array | float,
) -> Telescope:
    """Scale curvatures for mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        scale_factors: Scale factors per mirror (N,) or single value

    Returns:
        New Telescope with scaled mirror curvatures
    """
    new_curvatures = telescope.mirror_groups[group_idx].curvatures * jnp.asarray(scale_factors)
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.curvatures, new_curvatures
    )


def offset_mirror_curvatures(
    telescope,
    group_idx: int,
    offsets: Array | float,
) -> Telescope:
    """Add offset to curvatures for mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        offsets: Offsets per mirror (N,) or single value to add

    Returns:
        New Telescope with offset mirror curvatures
    """
    new_curvatures = telescope.mirror_groups[group_idx].curvatures + jnp.asarray(offsets)
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.curvatures, new_curvatures
    )


def set_focal_lengths(
    telescope,
    group_idx: int,
    focal_lengths: Array,
) -> Telescope:
    """Set mirror curvatures to achieve target focal lengths.

    For spherical/parabolic mirrors: c = 1/(2f).

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        focal_lengths: Target focal lengths array (N,)

    Returns:
        New Telescope with curvatures set for target focal lengths
    """
    focal_lengths = jnp.asarray(focal_lengths)
    # c = 1/(2f), handle infinite focal length (flat mirror) as zero curvature
    new_curvatures = jnp.where(
        jnp.isinf(focal_lengths),
        0.0,
        1.0 / (2.0 * focal_lengths)
    )
    return set_mirror_curvatures(telescope, group_idx, new_curvatures)


def apply_conic_error_to_group(
    telescope,
    group_idx: int,
    sigma: float,
    key: Array,
) -> Telescope:
    """Apply random Gaussian error to mirror conic constants.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        sigma: Standard deviation of conic constant error
        key: JAX random key for reproducibility

    Returns:
        New Telescope with perturbed mirror conic constants
    """
    group = telescope.mirror_groups[group_idx]
    noise = jax.random.normal(key, shape=(len(group),))
    new_conics = group.conics + noise * sigma
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.conics, new_conics
    )


def apply_aspheric_error_to_group(
    telescope,
    group_idx: int,
    sigmas: Array,
    key: Array,
) -> Telescope:
    """Apply random Gaussian errors to mirror aspheric coefficients.

    Each aspheric term can have its own sigma value.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        sigmas: Standard deviations per aspheric term (K,) where K is number of terms.
                If fewer sigmas than terms, remaining terms get zero error.
                If more sigmas than terms, extra sigmas are ignored.
        key: JAX random key for reproducibility

    Returns:
        New Telescope with perturbed mirror aspheric coefficients
    """
    group = telescope.mirror_groups[group_idx]
    n_mirrors = len(group)
    n_terms = group.aspherics.shape[1]

    sigmas = jnp.asarray(sigmas)
    # Pad or truncate sigmas to match number of terms
    if sigmas.size < n_terms:
        sigmas = jnp.concatenate([sigmas, jnp.zeros(n_terms - sigmas.size)])
    else:
        sigmas = sigmas[:n_terms]

    # Generate noise for each mirror and each term: (N, K)
    noise = jax.random.normal(key, shape=(n_mirrors, n_terms))
    # Scale by per-term sigmas
    perturbations = noise * sigmas[None, :]

    new_aspherics = group.aspherics + perturbations
    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.aspherics, new_aspherics
    )


def apply_focal_error_to_group(
    telescope,
    group_idx: int,
    sigma: float,
    key: Array,
    relative: bool = False,
) -> Telescope:
    """Apply random Gaussian error to mirror focal lengths.

    Perturbs the focal length of each mirror and converts back to curvature.
    For spherical/parabolic mirrors: f = 1/(2c), c = 1/(2f).

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group to modify
        sigma: Error magnitude:
            - If relative=True: fractional error (e.g., 0.01 for 1%)
            - If relative=False: absolute error in same units as focal length
        key: JAX random key for reproducibility
        relative: If True, apply relative (percentage) error; if False, absolute

    Returns:
        New Telescope with perturbed mirror curvatures
    """
    group = telescope.mirror_groups[group_idx]
    curvatures = group.curvatures

    # Convert curvature to focal length: f = 1/(2c)
    # Handle zero curvature (flat mirrors) by using a large focal length
    safe_curvatures = jnp.where(curvatures == 0, 1e-10, curvatures)
    focal_lengths = 1.0 / (2.0 * safe_curvatures)

    # Generate random perturbations
    noise = jax.random.normal(key, shape=(len(group),))

    if relative:
        # Relative error: f_new = f * (1 + noise * sigma)
        new_focal_lengths = focal_lengths * (1.0 + noise * sigma)
    else:
        # Absolute error: f_new = f + noise * sigma
        new_focal_lengths = focal_lengths + noise * sigma

    # Convert back to curvature: c = 1/(2f)
    # Preserve zero curvature for originally flat mirrors
    new_curvatures = jnp.where(
        curvatures == 0,
        0.0,
        1.0 / (2.0 * new_focal_lengths)
    )

    return _update_mirror_group_attr(
        telescope, group_idx, lambda g: g.curvatures, new_curvatures
    )


def get_mirrors_by_stage(telescope: Telescope, stage: int) -> list[int]:
    """Get indices of mirror groups at a specific optical stage.

    Args:
        telescope: Telescope instance
        stage: Optical stage (0=primary, 1=secondary, etc.)

    Returns:
        List of mirror group indices at the specified stage
    """
    return [
        i for i, g in enumerate(telescope.mirror_groups) if g.optical_stage == stage
    ]


def get_mirror_count(telescope: Telescope) -> int:
    """Get total number of mirrors across all groups.

    Args:
        telescope: Telescope instance

    Returns:
        Total mirror count
    """
    return sum(len(g) for g in telescope.mirror_groups)


# Sensor Operations


def add_sensor(telescope: Telescope, sensor: SensorGroup) -> Telescope:
    """Add a new sensor group to the telescope.

    Args:
        telescope: Telescope instance
        sensor: SensorGroup to add

    Returns:
        New Telescope with added sensor group
    """
    new_sensors = list(telescope.sensors) + [sensor]
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def replace_sensor(telescope: Telescope, sensor: SensorGroup, idx: int = 0) -> Telescope:
    """Replace sensor group by index.

    Args:
        telescope: Telescope instance
        sensor: SensorGroup replacement
        idx: Index of sensor group to replace (default: 0)

    Returns:
        New telescope with replaced sensor group

    Raises:
        IndexError: If index is out of range
    """
    if idx < 0 or idx >= len(telescope.sensors):
        raise IndexError(
            f"Sensor index {idx} out of range (0-{len(telescope.sensors)-1})"
        )
    new_sensors = list(telescope.sensors)
    new_sensors[idx] = sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def remove_sensor(telescope: Telescope, idx: int = 0) -> Telescope:
    """Remove a sensor group by index.

    Args:
        telescope: Telescope instance
        idx: Index of sensor group to remove (default: 0)

    Returns:
        New Telescope with sensor group removed

    Raises:
        IndexError: If idx is out of range
    """
    if idx < 0 or idx >= len(telescope.sensors):
        raise IndexError(
            f"Sensor index {idx} out of range (0-{len(telescope.sensors)-1})"
        )
    new_sensors = [s for i, s in enumerate(telescope.sensors) if i != idx]
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def set_sensor_positions(
    telescope: Telescope, group_idx: int, positions: Array
) -> Telescope:
    """Set positions for all sensors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of sensor group
        positions: New positions array (N, 3)

    Returns:
        New Telescope with updated sensor positions
    """
    return _update_sensor_attr(
        telescope, group_idx, lambda s: s.positions, jnp.asarray(positions)
    )


def set_sensor_rotations(
    telescope: Telescope, group_idx: int, rotations: Array
) -> Telescope:
    """Set rotations for all sensors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of sensor group
        rotations: New rotations array (N, 3) Euler angles in degrees

    Returns:
        New Telescope with updated sensor rotations
    """
    return _update_sensor_attr(
        telescope, group_idx, lambda s: s.rotations, jnp.asarray(rotations)
    )


def focus(telescope: Telescope, delta_z: float, sensor_idx: int = 0) -> Telescope:
    """Adjust all sensor positions in a group along optical axis (z-axis) for focus.

    Args:
        telescope: Telescope instance
        delta_z: Distance to move sensors along z-axis (positive = away from mirrors)
        sensor_idx: Index of sensor group to adjust (default: 0)

    Returns:
        New Telescope with adjusted sensor positions
    """
    current_positions = telescope.sensors[sensor_idx].positions
    new_positions = current_positions.at[:, 2].add(delta_z)
    return set_sensor_positions(telescope, sensor_idx, new_positions)


def get_sensor_count(telescope: Telescope) -> int:
    """Get number of sensor groups.

    Args:
        telescope: Telescope instance

    Returns:
        Number of sensor groups
    """
    return len(telescope.sensors)


def get_total_sensor_count(telescope: Telescope) -> int:
    """Get total number of individual sensors across all groups.

    Args:
        telescope: Telescope instance

    Returns:
        Total number of sensors
    """
    return sum(len(s) for s in telescope.sensors)


def with_ste(telescope: Telescope, sensor_idx: int = 0) -> Telescope:
    """Convert sensor group to straight-through estimator variant.

    Returns a new telescope with the specified sensor group converted to use
    straight-through estimation: hard assignment in forward pass,
    differentiable interpolation (bilinear for square, barycentric for hex)
    in backward pass.

    Args:
        telescope: Telescope instance
        sensor_idx: Index of sensor group to convert

    Returns:
        New Telescope with converted sensor group

    Raises:
        IndexError: If sensor_idx is out of range
        TypeError: If sensor type is not supported for conversion
    """
    from ..sensors import (
        HexagonalSensorGroup,
        SquareSensorGroup,
        StraightThroughHexagonalSensorGroup,
        StraightThroughSquareSensorGroup,
    )

    if sensor_idx < 0 or sensor_idx >= len(telescope.sensors):
        raise IndexError(
            f"Sensor index {sensor_idx} out of range (0-{len(telescope.sensors)-1})"
        )

    sensor = telescope.sensors[sensor_idx]

    new_sensor: StraightThroughSquareSensorGroup | StraightThroughHexagonalSensorGroup
    # Convert based on sensor type
    if isinstance(sensor, SquareSensorGroup) and not isinstance(
        sensor, StraightThroughSquareSensorGroup
    ):
        new_sensor = StraightThroughSquareSensorGroup(
            positions=sensor.positions,
            rotations=sensor.rotations,
            width=sensor.width,
            height=sensor.height,
            bounds=(
                sensor.x0,
                sensor.x0 + sensor.width * sensor.dx,
                sensor.y0,
                sensor.y0 + sensor.height * sensor.dy,
            ),
            edge_width=sensor.edge_width,
        )
    elif isinstance(sensor, HexagonalSensorGroup) and not isinstance(
        sensor, StraightThroughHexagonalSensorGroup
    ):
        new_sensor = StraightThroughHexagonalSensorGroup(
            positions=sensor.positions,
            rotations=sensor.rotations,
            hex_centers=sensor.hex_centers,
            edge_width=sensor.edge_width,
        )
    elif isinstance(
        sensor, StraightThroughSquareSensorGroup | StraightThroughHexagonalSensorGroup
    ):
        # Already a straight-through sensor, return unchanged
        return telescope
    else:
        raise TypeError(
            f"Cannot convert sensor type {type(sensor).__name__} to straight-through estimator"
        )

    new_sensors = list(telescope.sensors)
    new_sensors[sensor_idx] = new_sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


# Obstruction Operations


def add_obstruction(
    telescope: Telescope, obstruction: ObstructionGroup
) -> Telescope:
    """Add an obstruction group

    Args:
        telescope: Telescope instance
        obstruction: Obstruction group to add to telescope

    Returns:
        New Telescope with obstruction group added
    """
    current = telescope.obstruction_groups or []
    new_groups = list(current) + [obstruction]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def remove_obstruction(telescope: Telescope, group_idx: int) -> Telescope:
    """Remove an obstruction group by index.

    Args:
        telescope: Telescope instance
        group_idx: Index of obstruction group to remove

    Returns:
        New Telescope with obstruction group removed

    Raises:
        IndexError: If group_idx is out of range
    """
    if not telescope.obstruction_groups:
        raise IndexError("No obstruction groups to remove")
    if group_idx < 0 or group_idx >= len(telescope.obstruction_groups):
        raise IndexError(
            f"Obstruction group index {group_idx} out of range "
            f"(0-{len(telescope.obstruction_groups)-1})"
        )
    new_groups = [
        g for i, g in enumerate(telescope.obstruction_groups) if i != group_idx
    ]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def clear_obstructions(telescope: Telescope) -> Telescope:
    """Remove all obstructions from telescope.

    Args:
        telescope: Telescope instance

    Returns:
        New Telescope with no obstructions
    """
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, [])


def get_obstruction_count(telescope: Telescope) -> int:
    """Get total number of obstructions across all groups.

    Args:
        telescope: Telescope instance

    Returns:
        Total obstruction count
    """
    if not telescope.obstruction_groups:
        return 0
    return sum(len(g) for g in telescope.obstruction_groups)


# Convenience Operations


def clone(telescope: Telescope) -> Telescope:
    """Create a deep copy of the telescope.

    Args:
        telescope: Telescope instance

    Returns:
        Independent copy of the telescope
    """
    return jax.tree_util.tree_map(lambda x: x, telescope)


def get_info(telescope: Telescope) -> dict[str, Any]:
    """Get summary information about telescope configuration.

    Args:
        telescope: Telescope instance

    Returns:
        Dictionary with telescope statistics and properties
    """
    from ..sensors import HexagonalSensorGroup, SquareSensorGroup
    from .mirrors import AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup

    # Mirror info
    n_mirror_groups = len(telescope.mirror_groups)
    n_mirrors = get_mirror_count(telescope)

    stages: set[int] = set()
    mirror_types: list[str] = []
    for group in telescope.mirror_groups:
        stages.add(group.optical_stage)
        if isinstance(group, AsphericDiskMirrorGroup):
            mirror_types.append("disk")
        elif isinstance(group, AsphericPolygonMirrorGroup):
            mirror_types.append("polygon")
        else:
            mirror_types.append("unknown")

    # Sensor info
    n_sensor_groups = len(telescope.sensors)
    n_sensors_total = get_total_sensor_count(telescope)
    sensor_types: list[str] = []
    sensors_per_group: list[int] = []
    for sensor in telescope.sensors:
        sensors_per_group.append(len(sensor))
        if isinstance(sensor, SquareSensorGroup):
            sensor_types.append("square")
        elif isinstance(sensor, HexagonalSensorGroup):
            sensor_types.append("hexagonal")
        else:
            sensor_types.append(type(sensor).__name__)

    # Obstruction info
    n_obstruction_groups = (
        len(telescope.obstruction_groups) if telescope.obstruction_groups else 0
    )
    n_obstructions = get_obstruction_count(telescope)

    # Compute bounding box of mirrors
    if telescope.mirror_groups:
        all_positions = jnp.concatenate(
            [g.positions for g in telescope.mirror_groups], axis=0
        )
        bbox_min = all_positions.min(axis=0)
        bbox_max = all_positions.max(axis=0)
    else:
        bbox_min = bbox_max = jnp.zeros(3)

    return {
        "name": telescope.name,
        "n_mirror_groups": n_mirror_groups,
        "n_mirrors": n_mirrors,
        "optical_stages": sorted(stages),
        "mirror_types": mirror_types,
        "n_sensor_groups": n_sensor_groups,
        "n_sensors_total": n_sensors_total,
        "sensors_per_group": sensors_per_group,
        "sensor_types": sensor_types,
        "n_obstruction_groups": n_obstruction_groups,
        "n_obstructions": n_obstructions,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }
