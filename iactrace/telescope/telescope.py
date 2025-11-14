"""Telescope configuration and compilation."""

import jax
import jax.numpy as jnp
import equinox as eqx
import yaml

from .apertures import DiskAperture, PolygonAperture
from .surfaces import AsphericSurface

class Telescope(eqx.Module):
    """
    IACT telescope configuration as an Equinox Module (PyTree).
    
    All data needed for rendering is stored as fields and can be directly
    passed to JAX-compiled render functions.
    """
    
    # Mirror geometry
    mirror_positions: jax.Array  # (N, 3)
    mirror_rotations: jax.Array  # (N, 3)
    mirror_points: jax.Array  # (N, M, 3) - sampled surface points
    mirror_normals: jax.Array  # (N, M, 3) - surface normals
    mirror_surfaces: list
    mirror_apertures: list
    
    # Sensor plane geometry
    sensor_positions: jax.Array  # (L, 3)
    sensor_rotations: jax.Array  # (L, 3)
    sensor_configs: list = eqx.field(static=True)
    
    # Obstructions
    cyl_p1: jax.Array  # (C, 3)
    cyl_p2: jax.Array  # (C, 3)
    cyl_r: jax.Array  # (C,)
    box_p1: jax.Array  # (B, 3)
    box_p2: jax.Array  # (B, 3)
    
    # Metadata 
    name: str = eqx.field(static=True)
    
    # Define which parameters are frozen:
    _frozen: frozenset

    def __init__(self, mirror_positions, mirror_rotations,
                 mirror_points, mirror_normals,
                 mirror_surfaces, mirror_apertures,
                 sensor_positions, sensor_rotations,
                 sensor_configs,
                 cyl_p1=None, cyl_p2=None, cyl_r=None,
                 box_p1=None, box_p2=None,
                 name="telescope", frozen=None):
        """
        Initialize Telescope with all required data.
        
        Args:
            mirror_positions: Mirror center positions (N, 3)
            mirror_rotations: Mirror rotation euler angles (N, 3)
            mirror_points: Sampled points on mirror surfaces (N, M, 3)
            mirror_normals: Normals at sampled points (N, M, 3)
            cyl_p1, cyl_p2, cyl_radius: Cylinder obstructions
            box_p1, box_p2: Box obstructions
            sensor_positions: Sensor plane positions (L, 3)
            sensor_rotations: Sensor plane euler angles (L, 3)
            mirror_templates: List of mirror surface definitions
            mirror_apertures: List of aperture definitions
            sensor_configs: Sensor configuration dict
            name: Telescope name
        """
        # Mirror positions and sampled points
        self.mirror_positions = jnp.array(mirror_positions)
        self.mirror_rotations = jnp.array(mirror_rotations)
        self.mirror_points = jnp.array(mirror_points)
        self.mirror_normals = jnp.array(mirror_normals)

        # Sensor plane positions and rotations
        self.sensor_positions = jnp.array(sensor_positions)
        self.sensor_rotations = jnp.array(sensor_rotations)
        
        # Obstructions (default to empty)
        self.cyl_p1 = jnp.array(cyl_p1) if cyl_p1 is not None else jnp.zeros((0, 3))
        self.cyl_p2 = jnp.array(cyl_p2) if cyl_p2 is not None else jnp.zeros((0, 3))
        self.cyl_r = jnp.array(cyl_r) if cyl_r is not None else jnp.zeros((0,))
        self.box_p1 = jnp.array(box_p1) if box_p1 is not None else jnp.zeros((0, 3))
        self.box_p2 = jnp.array(box_p2) if box_p2 is not None else jnp.zeros((0, 3))
        
        # Static metadata
        self.mirror_surfaces = mirror_surfaces
        self.mirror_apertures = mirror_apertures
        self.sensor_configs = sensor_configs
        self.name = name
        
        # Standard setting is to freeze all parameters:
        if frozen is None:
            array_fields = frozenset(
                k for k, v in self.__dict__.items() 
                if isinstance(v, jax.Array)
            )
            self._frozen = array_fields
        else:
            self._frozen = frozen

    @classmethod
    def from_yaml(cls, filename, integrator, sampling_key=None):
        """
        Load and compile telescope from YAML file.
        
        Args:
            filename: Path to YAML configuration file
            integrator: Integrator object for sampling mirror surfaces
            sampling_key: JAX random key for sampling
        
        Returns:
            Telescope object
        """
        if sampling_key is None:
            sampling_key = jax.random.key(0)
        
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        telescope_name = config.get('telescope', {}).get('name', 'telescope')
        
        # Parse mirrors
        mirrors = config.get('mirrors', [])
        templates = config.get('mirror_templates', {})
        
        mirror_positions = []
        mirror_rotations = []
        mirror_apertures = []
        mirror_surfaces = []
        
        for mirror in mirrors:
            # Position
            mirror_positions.append(mirror['position'])
            mirror_rotations.append(mirror['orientation'])
            # Aperture
            if mirror['aperture']['type'] == 'circular':
                mirror_apertures.append(DiskAperture(mirror['aperture']['radius']))
            elif mirror['aperture']['type'] == 'polygon':
                mirror_apertures.append(PolygonAperture(mirror['aperture']['vertices']))
            # Surface
            mirror_surfaces.append(AsphericSurface.from_template(templates[mirror['template']]))
            
        mirror_positions = jnp.array(mirror_positions)
        mirror_rotations = jnp.array(mirror_rotations)
        
        # Sample mirrors:
        mirror_points, mirror_normals = integrator.sample(
            mirror_surfaces, mirror_apertures, sampling_key
        )

        # Parse obstructions
        cyl_p1_list, cyl_p2_list, cyl_r_list = [], [], []
        box_p1_list, box_p2_list = [], []
        
        for obs in config.get('obstructions', []):
            obs_type = obs['type']
            
            if obs_type == 'cylinder':
                cyl_p1_list.append(obs['p1'])
                cyl_p2_list.append(obs['p2'])
                cyl_r_list.append(obs['r'])
        
            elif obs_type == 'box':
                box_p1_list.append(obs['p1'])
                box_p2_list.append(obs['p2'])
        
        cyl_p1 = jnp.array(cyl_p1_list) if cyl_p1_list else None
        cyl_p2 = jnp.array(cyl_p2_list) if cyl_p2_list else None
        cyl_r = jnp.array(cyl_r_list) if cyl_r_list else None
        box_p1 = jnp.array(box_p1_list) if box_p1_list else None
        box_p2 = jnp.array(box_p2_list) if box_p2_list else None
        
        # Parse sensors
        sensors = config.get('sensors', [])
        sensor_positions = []
        sensor_rotations = []
        sensor_configs = []
        
        for sensor in sensors:
            sensor_positions.append(sensor['position'])
            sensor_rotations.append(sensor['orientation'])
            sensor_configs.append(sensor)

        # Return Object:
        return cls(
            mirror_positions=mirror_positions,
            mirror_rotations=mirror_rotations,
            mirror_points=mirror_points,
            mirror_normals=mirror_normals,
            mirror_surfaces=mirror_surfaces,
            mirror_apertures=mirror_apertures,
            cyl_p1=cyl_p1,
            cyl_p2=cyl_p2,
            cyl_r=cyl_r,
            box_p1=box_p1,
            box_p2=box_p2,
            sensor_positions=sensor_positions,
            sensor_rotations=sensor_rotations,
            sensor_configs=sensor_configs,
            name=telescope_name
        )
    
    # Simple methods
    def freeze(self, *field_names):
        """Freeze fields (make non-trainable)"""
        return eqx.tree_at(lambda t: t._frozen, self, 
                          self._frozen | frozenset(field_names))
    
    def unfreeze(self, *field_names):
        """Unfreeze fields (make trainable)"""
        return eqx.tree_at(lambda t: t._frozen, self,
                          self._frozen - frozenset(field_names))
    
    def freeze_all(self):
        """Freeze all array fields"""
        fields = {k for k, v in vars(self).items() if isinstance(v, jax.Array)}
        return self.freeze(*fields)
    
    def with_trainable(self, *field_names):
        """Set exactly these fields as trainable, freeze rest"""
        fields = {k for k, v in vars(self).items() if isinstance(v, jax.Array)}
        frozen = fields - set(field_names)
        return eqx.tree_at(lambda t: t._frozen, self, frozenset(frozen))
    
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