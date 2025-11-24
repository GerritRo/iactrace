import jax
import jax.numpy as jnp
import equinox as eqx
import dataclasses
import yaml

from .apertures import DiskAperture, PolygonAperture
from .surfaces import AsphericSurface
from ..core import render, render_debug
from ..sensors import SquareSensor, HexagonalSensor

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
    mirror_weights: jax.Array # (N, M, 1) - sampling weights
    mirror_surfaces: list
    mirror_apertures: list
    
    # Obstructions
    cyl_p1: jax.Array  # (C, 3)
    cyl_p2: jax.Array  # (C, 3)
    cyl_r: jax.Array  # (C,)
    box_p1: jax.Array  # (B, 3)
    box_p2: jax.Array  # (B, 3)
    
    # Sensors
    sensors: list
    
    # Metadata 
    name: str = eqx.field(static=True)
    
    # Define which parameters are frozen:
    _frozen: frozenset = eqx.field(static=True)

    def __init__(self, mirror_positions, mirror_rotations,
                 mirror_points, mirror_normals, mirror_weights,
                 mirror_surfaces, mirror_apertures,
                 sensors,
                 cyl_p1=None, cyl_p2=None, cyl_r=None,
                 box_p1=None, box_p2=None,
                 name="telescope", _frozen=None):
        """
        Initialize Telescope with all required data.
        
        Args:
            mirror_positions: Mirror center positions (N, 3)
            mirror_rotations: Mirror rotation euler angles (N, 3)
            mirror_points: Sampled points on mirror surfaces (N, M, 3)
            mirror_normals: Normals at sampled points (N, M, 3)
            mirror_weights: Cos angle to z axis at sampling
            cyl_p1, cyl_p2, cyl_radius: Cylinder obstructions
            box_p1, box_p2: Box obstructions
            sensors: Sensor objects
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
        self.mirror_weights = jnp.array(mirror_weights)
        self.mirror_surfaces = mirror_surfaces
        self.mirror_apertures = mirror_apertures
        
        # Obstructions (default to empty)
        self.cyl_p1 = jnp.array(cyl_p1) if cyl_p1 is not None else jnp.zeros((0, 3))
        self.cyl_p2 = jnp.array(cyl_p2) if cyl_p2 is not None else jnp.zeros((0, 3))
        self.cyl_r = jnp.array(cyl_r) if cyl_r is not None else jnp.zeros((0,))
        self.box_p1 = jnp.array(box_p1) if box_p1 is not None else jnp.zeros((0, 3))
        self.box_p2 = jnp.array(box_p2) if box_p2 is not None else jnp.zeros((0, 3))
        
        # Static metadata
        self.sensors = sensors
        self.name = name
        
        # Standard setting is to freeze all parameters:
        if _frozen is None:
            array_fields = frozenset(
                k for k, v in self.__dict__.items() 
                if isinstance(v, jax.Array)
            )
            self._frozen = array_fields
        else:
            self._frozen = _frozen

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
        mirror_points, mirror_normals, mirror_weights = integrator.sample(
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
        sensors = []
        
        for sensor_data in config.get('sensors', []):
            sensor_type = sensor_data['type']
            if sensor_type == 'square':
                sensor = SquareSensor(position=sensor_data['position'],
                                      rotation=sensor_data['orientation'],
                                      width=sensor_data['width'],
                                      height=sensor_data['height'],
                                      bounds=tuple(sensor_data['bounds']))
            elif sensor_type == 'hexagonal':
                pixel_centers = jnp.array([sensor_data['centers_x'], sensor_data['centers_y']]).T
                sensor = HexagonalSensor(position=sensor_data['position'],
                                         rotation=sensor_data['orientation'],
                                         hex_centers=pixel_centers)
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
            sensors.append(sensor)

        # Return Object:
        return cls(
            mirror_positions=mirror_positions,
            mirror_rotations=mirror_rotations,
            mirror_points=mirror_points,
            mirror_normals=mirror_normals,
            mirror_weights=mirror_weights,
            mirror_surfaces=mirror_surfaces,
            mirror_apertures=mirror_apertures,
            sensors=sensors,
            cyl_p1=cyl_p1,
            cyl_p2=cyl_p2,
            cyl_r=cyl_r,
            box_p1=box_p1,
            box_p2=box_p2,
            name=telescope_name
        )
    
    def __call__(self, sources, values, source_type='point', sensor_idx=0, debug=False, **overrides):
        """
        Render sources through telescope.

        Args:
            sources: Source positions (N, 3) or directions (N, 3)
            values: Flux of the source in ph/m^2 (N, )
            source_type: 'point' or 'infinity'
            sensor_idx: Which sensor to use (default 0)
            **overrides: Override fields (e.g., mirror_normals=new_normals)

        Returns:
            Rendered image
        """
        # Apply overrides if provided
        telescope = self
        if overrides:
            for key, value in overrides.items():
                telescope = eqx.tree_at(lambda t: getattr(t, key), telescope, value)

        if debug == False:
            return render(telescope, sources, values, source_type, sensor_idx)
        else:
            return render_debug(telescope, sources, values, source_type, sensor_idx)
    
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