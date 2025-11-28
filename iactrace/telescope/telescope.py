import jax
import jax.numpy as jnp
import equinox as eqx
import yaml

from typing import Callable
from .mirrors import Mirror, group_mirrors
from .apertures import DiskAperture, PolygonAperture
from .surfaces import AsphericSurface
from .obstructions import Cylinder, Box, group_obstructions
from ..core import render, render_debug
from ..sensors import SquareSensor, HexagonalSensor


class Telescope(eqx.Module):
    """
    IACT telescope configuration as an Equinox Module.

    Stores mirrors, obstructions, and sensors as object lists for
    polymorphic dispatch while maintaining JAX compatibility.
    """

    mirror_groups: list             # Grouped by surface/aperture for fast rendering
    obstruction_groups: list        # Grouped by type for fast rendering
    sensors: list
    name: str = eqx.field(static=True)

    def __init__(self, mirror_groups, obstruction_groups=None, sensors=None, name="telescope"):
        """
        Initialize Telescope.

        Args:
            mirror_groups: List of Mirror groups
            obstructions: List of Obstruction groups
            sensors: List of sensor objects
            name: Telescope name
        """
        self.mirror_groups = mirror_groups
        self.obstruction_groups = obstruction_groups
        self.sensors = list(sensors) if sensors else []
        self.name = name
        
    @classmethod
    def from_yaml(cls, filename, integrator, sampling_key=None):
        """
        Load telescope from YAML configuration file.
        
        Args:
            filename: Path to YAML file
            integrator: Integrator for sampling mirror surfaces
            sampling_key: JAX random key
        
        Returns:
            Telescope object with sampled mirrors
        """
        if sampling_key is None:
            sampling_key = jax.random.key(0)
        
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        telescope_name = config.get('telescope', {}).get('name', 'telescope')
        templates = config.get('mirror_templates', {})
        
        # Parse mirrors
        mirrors = []
        for m in config.get('mirrors', []):
            # Aperture
            if m['aperture']['type'] == 'circular':
                aperture = DiskAperture(m['aperture']['radius'])
            elif m['aperture']['type'] == 'polygon':
                aperture = PolygonAperture(m['aperture']['vertices'])
            else:
                raise ValueError(f"Unknown aperture type: {m['aperture']['type']}")
            
            # Surface
            surface = AsphericSurface.from_template(templates[m['template']])
            
            mirrors.append(Mirror(
                position=m['position'],
                rotation=m['orientation'],
                surface=surface,
                aperture=aperture
            ))
        
        # Create and sample mirror groups
        mirror_groups = group_mirrors(mirrors)
        mirror_groups = integrator.sample_mirror_groups(mirror_groups, sampling_key)
        
        # Parse obstructions
        obstructions = []
        for obs in config.get('obstructions', []):
            if obs['type'] == 'cylinder':
                obstructions.append(Cylinder(obs['p1'], obs['p2'], obs['r']))
            elif obs['type'] == 'box':
                obstructions.append(Box(obs['p1'], obs['p2']))
            else:
                raise ValueError(f"Unknown obstruction type: {obs['type']}")
        
        # Create obstruction groups
        obstruction_groups = group_obstructions(obstructions)
        
        # Parse sensors
        sensors = []
        for s in config.get('sensors', []):
            if s['type'] == 'square':
                sensors.append(SquareSensor(
                    position=s['position'],
                    rotation=s['orientation'],
                    width=s['width'],
                    height=s['height'],
                    bounds=tuple(s['bounds'])
                ))
            elif s['type'] == 'hexagonal':
                centers = jnp.array([s['centers_x'], s['centers_y']]).T
                sensors.append(HexagonalSensor(
                    position=s['position'],
                    rotation=s['orientation'],
                    hex_centers=centers
                ))
            else:
                raise ValueError(f"Unknown sensor type: {s['type']}")
        
        return cls(
            mirror_groups=mirror_groups,
            obstruction_groups=obstruction_groups,
            sensors=sensors,
            name=telescope_name
        )
    
    def __call__(self, sources, values, source_type='point', sensor_idx=0, debug=False):
        """
        Render sources through telescope.
        
        Args:
            sources: Source positions (N, 3) or directions (N, 3)
            values: Flux values (N,)
            source_type: 'point' or 'infinity'
            sensor_idx: Which sensor to use
            debug: If True, return raw hits instead of accumulated image
        
        Returns:
            Rendered image or (pts, values) if debug=True
        """
        if debug:
            return render_debug(self, sources, values, source_type, sensor_idx)
        return render(self, sources, values, source_type, sensor_idx)
    
    def resample(self, integrator, key):
        """Return new telescope with resampled mirror groups."""
        new_mirror_groups = integrator.sample_mirror_groups(self.mirror_groups, key)
        return eqx.tree_at(
            lambda t: t.mirror_groups,
            self,
            new_mirror_groups
        )
    
    def freeze(self, *field_names):
        """Freeze fields (make non-trainable)"""
        return dataclasses.replace(
            self,
            _frozen=self._frozen | frozenset(field_names)
        )
    
    def unfreeze(self, *field_names):
        """Unfreeze fields (make trainable)"""
        return dataclasses.replace(
            self,
            _frozen=self._frozen - frozenset(field_names)
        )
    
    def freeze_all(self):
        """Freeze all array fields"""
        fields = frozenset(
            k for k, v in vars(self).items() 
            if isinstance(v, jax.Array)
        )
        return dataclasses.replace(self, _frozen=fields)
    
    def with_trainable(self, *field_names):
        """Set exactly these fields as trainable, freeze rest"""
        all_fields = frozenset(
            k for k, v in vars(self).items() 
            if isinstance(v, jax.Array)
        )
        new_frozen = all_fields - frozenset(field_names)
        return dataclasses.replace(self, _frozen=new_frozen)
    
    def is_frozen(self, field_name):
        """Check if field is frozen"""
        return field_name in self._frozen
    
    def partition_trainable(self):
        """Split into trainable and frozen parts"""
        def is_trainable(path, leaf):
            return isinstance(leaf, jax.Array) and path[-1] not in self._frozen
        return eqx.partition(self, is_trainable)
    
    @property
    def trainable_fields(self):
        """Get list of trainable fields"""
        return [k for k, v in vars(self).items() 
                if isinstance(v, jax.Array) and k not in self._frozen]
    
    @property
    def frozen_fields(self):
        """Get list of frozen fields"""
        return list(self._frozen)