import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Dict, Any, Union, List

from ..core import euler_to_matrix

# Mirror operations

def resample_mirrors(telescope, integrator, key):
    """Resample all mirrors with specified integrator
    
    Args:
        telescope: Telescope instance
        integrator: Integrator object
        key: jax.random.key
        
    Returns:
        New telescope with resampled mirrors
    """
    keys = jax.random.split(key, len(telescope.mirror_groups))
    new_groups = [integrator.sample_group(g, k) 
                  for g, k in zip(telescope.mirror_groups, keys)]
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def set_mirror_positions(telescope, group_idx: int, positions):
    """Set positions for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        positions: New positions array (N, 3)

    Returns:
        New Telescope with updated mirror positions
    """
    positions = jnp.asarray(positions)
    new_group = eqx.tree_at(
        lambda g: g.positions,
        telescope.mirror_groups[group_idx],
        positions
    )
    new_groups = list(telescope.mirror_groups)
    new_groups[group_idx] = new_group
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def set_mirror_rotations(telescope, group_idx: int, rotations):
    """Set rotations for all mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        rotations: New rotations array (N, 3) in degrees (Euler angles)

    Returns:
        New Telescope with updated mirror rotations
    """
    rotations = jnp.asarray(rotations)
    new_group = eqx.tree_at(
        lambda g: g.rotations,
        telescope.mirror_groups[group_idx],
        rotations
    )
    new_groups = list(telescope.mirror_groups)
    new_groups[group_idx] = new_group
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def scale_mirror_weights(telescope, group_idx: int, scale_factors):
    """Scale reflectivity weights for mirrors in a group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        scale_factors: Scale factors per mirror (N,) or single value

    Returns:
        New Telescope with scaled mirror weights
    """
    scale_factors = jnp.asarray(scale_factors)
    if scale_factors.ndim == 0:
        scale_factors = jnp.full(len(telescope.mirror_groups[group_idx]), scale_factors)

    current_weights = telescope.mirror_groups[group_idx].weights
    # Scale factors shape (N,) -> (N, 1, 1) for broadcasting with (N, M, 1)
    new_weights = current_weights * scale_factors[:, None, None]

    new_group = eqx.tree_at(
        lambda g: g.weights,
        telescope.mirror_groups[group_idx],
        new_weights
    )
    new_groups = list(telescope.mirror_groups)
    new_groups[group_idx] = new_group
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def apply_roughness(telescope, roughness):
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
        new_groups.append(eqx.tree_at(lambda g: g.perturbation_scale, group, new_scale))
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def apply_roughness_to_group(telescope, group_idx: int, roughness: float):
    """Apply roughness to a specific mirror group.

    Args:
        telescope: Telescope instance
        group_idx: Index of mirror group
        roughness: Surface roughness in arcseconds

    Returns:
        New Telescope with updated roughness for specified group
    """
    sigma_rad = roughness * jnp.pi / (180.0 * 3600.0)
    group = telescope.mirror_groups[group_idx]
    new_scale = jnp.full(len(group), sigma_rad)
    new_group = eqx.tree_at(lambda g: g.perturbation_scale, group, new_scale)

    new_groups = list(telescope.mirror_groups)
    new_groups[group_idx] = new_group
    return eqx.tree_at(lambda t: t.mirror_groups, telescope, new_groups)


def get_mirrors_by_stage(telescope, stage: int) -> List[int]:
    """Get indices of mirror groups at a specific optical stage.

    Args:
        telescope: Telescope instance
        stage: Optical stage (0=primary, 1=secondary, etc.)

    Returns:
        List of mirror group indices at the specified stage
    """
    return [i for i, g in enumerate(telescope.mirror_groups) if g.optical_stage == stage]


def get_mirror_count(telescope) -> int:
    """Get total number of mirrors across all groups.

    Args:
        telescope: Telescope instance

    Returns:
        Total mirror count
    """
    return sum(len(g) for g in telescope.mirror_groups)


# Sensor Operations

def add_sensor(telescope, sensor):
    """Add a new sensor to the telescope.

    Args:
        telescope: Telescope instance
        sensor: Sensor to add

    Returns:
        New Telescope with added sensor
    """
    new_sensors = list(telescope.sensors) + [sensor]
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def replace_sensor(telescope, sensor, idx: int = 0):
    """Replace sensor by index.
    
    Args:
        telescope: Telescope instance
        sensor: Sensor replacement
        idx: Index of sensor to remove (default: 0)
        
    Returns:
        New telescope with replaced sensor
        
    Raises:
        IndexError: If index is out of range
    """
    if idx < 0 or idx >= len(telescope.sensors):
        raise IndexError(f"Sensor index {idx} out of range (0-{len(telescope.sensors)-1})")
    new_sensors = list(telescope.sensors)
    new_sensors[idx] = sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def remove_sensor(telescope, idx: int = 0):
    """Remove a sensor by index.

    Args:
        telescope: Telescope instance
        idx: Index of sensor to remove (default: 0)

    Returns:
        New Telescope with sensor removed

    Raises:
        IndexError: If idx is out of range
    """
    if idx < 0 or idx >= len(telescope.sensors):
        raise IndexError(f"Sensor index {idx} out of range (0-{len(telescope.sensors)-1})")
    new_sensors = [s for i, s in enumerate(telescope.sensors) if i != idx]
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def set_sensor_position(telescope, idx: int, position):
    """Set position of a sensor.

    Args:
        telescope: Telescope instance
        idx: Index of sensor
        position: New position (3,)

    Returns:
        New Telescope with updated sensor position
    """
    position = jnp.asarray(position)
    new_sensor = eqx.tree_at(lambda s: s.position, telescope.sensors[idx], position)
    new_sensors = list(telescope.sensors)
    new_sensors[idx] = new_sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def set_sensor_rotation(telescope, idx: int, rotation):
    """Set rotation of a sensor.

    Args:
        telescope: Telescope instance
        idx: Index of sensor
        rotation: New rotation (3,) Euler angles in degrees

    Returns:
        New Telescope with updated sensor rotation
    """
    rotation = jnp.asarray(rotation)
    new_sensor = eqx.tree_at(lambda s: s.rotation, telescope.sensors[idx], rotation)
    new_sensors = list(telescope.sensors)
    new_sensors[idx] = new_sensor
    return eqx.tree_at(lambda t: t.sensors, telescope, new_sensors)


def focus(telescope, delta_z: float, sensor_idx: int = 0):
    """Adjust sensor position along optical axis (z-axis) for focus.

    Args:
        telescope: Telescope instance
        delta_z: Distance to move sensor along z-axis (positive = away from mirrors)
        sensor_idx: Index of sensor to adjust (default: 0)

    Returns:
        New Telescope with adjusted sensor position
    """
    current_pos = telescope.sensors[sensor_idx].position
    new_pos = current_pos.at[2].add(delta_z)
    return set_sensor_position(telescope, sensor_idx, new_pos)


def get_sensor_count(telescope) -> int:
    """Get number of sensors.

    Args:
        telescope: Telescope instance

    Returns:
        Number of sensors
    """
    return len(telescope.sensors)


# Obstruction Operations

def add_obstruction(telescope, obstruction):
    """Add an obstruction group
    
    Args:
        telescope: Telescope instance
        obstruction: Obstruction group to add to telescope
        
    Returns:
        New Telescope with obstruction group added
    """
    new_groups = telescope.obstruction_groups + [obstruction]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def remove_obstruction(telescope, group_idx: int):
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
        raise IndexError(f"Obstruction group index {group_idx} out of range "
                        f"(0-{len(telescope.obstruction_groups)-1})")
    new_groups = [g for i, g in enumerate(telescope.obstruction_groups) if i != group_idx]
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, new_groups)


def clear_obstructions(telescope):
    """Remove all obstructions from telescope.

    Args:
        telescope: Telescope instance

    Returns:
        New Telescope with no obstructions
    """
    return eqx.tree_at(lambda t: t.obstruction_groups, telescope, [])


def get_obstruction_count(telescope) -> int:
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

def clone(telescope):
    """Create a deep copy of the telescope.

    Args:
        telescope: Telescope instance

    Returns:
        Independent copy of the telescope
    """
    return jax.tree_util.tree_map(lambda x: x, telescope)


def get_info(telescope) -> Dict[str, Any]:
    """Get summary information about telescope configuration.

    Args:
        telescope: Telescope instance

    Returns:
        Dictionary with telescope statistics and properties
    """
    from .mirrors import AsphericDiskMirrorGroup, AsphericPolygonMirrorGroup
    from ..sensors import SquareSensor, HexagonalSensor

    # Mirror info
    n_mirror_groups = len(telescope.mirror_groups)
    n_mirrors = get_mirror_count(telescope)

    stages = set()
    mirror_types = []
    for group in telescope.mirror_groups:
        stages.add(group.optical_stage)
        if isinstance(group, AsphericDiskMirrorGroup):
            mirror_types.append('disk')
        elif isinstance(group, AsphericPolygonMirrorGroup):
            mirror_types.append('polygon')
        else:
            mirror_types.append('unknown')

    # Sensor info
    n_sensors = len(telescope.sensors)
    sensor_types = []
    for sensor in telescope.sensors:
        if isinstance(sensor, SquareSensor):
            sensor_types.append('square')
        elif isinstance(sensor, HexagonalSensor):
            sensor_types.append('hexagonal')
        else:
            sensor_types.append(type(sensor).__name__)

    # Obstruction info
    n_obstruction_groups = len(telescope.obstruction_groups) if telescope.obstruction_groups else 0
    n_obstructions = get_obstruction_count(telescope)

    # Compute bounding box of mirrors
    if telescope.mirror_groups:
        all_positions = jnp.concatenate([g.positions for g in telescope.mirror_groups], axis=0)
        bbox_min = all_positions.min(axis=0)
        bbox_max = all_positions.max(axis=0)
    else:
        bbox_min = bbox_max = jnp.zeros(3)

    return {
        'name': telescope.name,
        'n_mirror_groups': n_mirror_groups,
        'n_mirrors': n_mirrors,
        'optical_stages': sorted(stages),
        'mirror_types': mirror_types,
        'n_sensors': n_sensors,
        'sensor_types': sensor_types,
        'n_obstruction_groups': n_obstruction_groups,
        'n_obstructions': n_obstructions,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
    }