import yaml
import jax
import jax.numpy as jnp

from ..telescope import Telescope, Mirror, group_mirrors
from ..core import (
    AsphericSurface,
    DiskAperture,
    PolygonAperture,
    Cylinder,
    Box,
    Sphere,
    OrientedBox,
    Triangle,
    group_obstructions,
)
from ..sensors import SquareSensor, HexagonalSensor


def load_telescope(filename, integrator, key=None):
    """
    Load telescope from YAML configuration file.
    
    Args:
        filename: Path to YAML file
        integrator: MCIntegrator for sampling mirrors
        key: JAX random key (default: key(0))
    
    Returns:
        Telescope
    """
    if key is None:
        key = jax.random.key(0)
    
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    
    return build_telescope(config, integrator, key)


def build_telescope(config, integrator, key):
    """
    Build telescope from parsed config dict.
    
    Args:
        config: Dict from YAML
        integrator: MCIntegrator
        key: JAX random key
    
    Returns:
        Telescope
    """
    name = config.get('telescope', {}).get('name', 'telescope')
    templates = config.get('mirror_templates', {})
    
    mirrors = [_parse_mirror(m, templates) for m in config.get('mirrors', [])]
    mirror_groups = group_mirrors(mirrors)
    
    # Only sample stage 0 (primary) mirrors
    sampled_groups = []
    for group in mirror_groups:
        if group.optical_stage == 0:
            key, subkey = jax.random.split(key)
            sampled_groups.append(integrator.sample_group(group, subkey))
        else:
            # Stage 1+ mirrors don't need sampling, keep as-is
            sampled_groups.append(group)
    
    obstructions = [_parse_obstruction(o) for o in config.get('obstructions', [])]
    obstruction_groups = group_obstructions(obstructions)
    
    sensors = [_parse_sensor(s) for s in config.get('sensors', [])]
    
    return Telescope(
        mirror_groups=sampled_groups,
        obstruction_groups=obstruction_groups,
        sensors=sensors,
        name=name,
    )


def _parse_mirror(m, templates):
    """Parse single mirror config."""
    aperture = _parse_aperture(m['aperture'])
    surface = AsphericSurface.from_template(templates[m['template']])
    
    # Default optical_stage to 0 if not specified
    optical_stage = m.get('stage', 0)
    # Default offset to [0, 0] if not specified
    offset = m.get('offset', [0.0, 0.0])
    
    return Mirror(
        position=m['position'],
        rotation=m['orientation'],
        surface=surface,
        aperture=aperture,
        optical_stage=optical_stage,
        offset=offset,
    )


def _parse_aperture(config):
    """Parse aperture config."""
    atype = config['type']
    
    if atype == 'circular':
        return DiskAperture(config['radius'])
    elif atype == 'polygon':
        return PolygonAperture(config['vertices'])
    else:
        raise ValueError(f"Unknown aperture type: {atype}")


def _parse_obstruction(config):
    """Parse obstruction config."""
    otype = config['type']
    
    if otype == 'cylinder':
        return Cylinder(config['p1'], config['p2'], config['r'])
    elif otype == 'box':
        return Box(config['p1'], config['p2'])
    elif otype == 'sphere':
        return Sphere(config['center'], config['r'])
    elif otype == 'oriented_box':
        return OrientedBox(
            config['center'],
            config['half_extents'],
            jnp.array(config['rotation']),
        )
    elif otype == 'triangle':
        return Triangle(config['v0'], config['v1'], config['v2'])
    else:
        raise ValueError(f"Unknown obstruction type: {otype}")


def _parse_sensor(config):
    """Parse sensor config."""
    stype = config['type']
    edge_width = config.get('edge_width', 0.0)
    
    if stype == 'square':
        return SquareSensor(
            position=config['position'],
            rotation=config['orientation'],
            width=config['width'],
            height=config['height'],
            bounds=tuple(config['bounds']),
            edge_width=edge_width,
        )
    elif stype == 'hexagonal':
        centers = jnp.array([config['centers_x'], config['centers_y']]).T
        return HexagonalSensor(
            position=config['position'],
            rotation=config['orientation'],
            hex_centers=centers,
            edge_width=edge_width,
        )
    else:
        raise ValueError(f"Unknown sensor type: {stype}")