import jax
import jax.numpy as jnp
import equinox as eqx
import yaml

from typing import Callable
from .mirrors import Mirror, group_mirrors
from ..core import DiskAperture, PolygonAperture
from ..core import AsphericSurface
from ..core import Cylinder, Box, group_obstructions
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

    @classmethod
    def from_yaml(cls, filename, integrator, key=None):
        """Load from YAML config."""
        from ..io.yaml_loader import load_telescope
        return load_telescope(filename, integrator, key)
    
    def resample(self, integrator, key):
        from .operations import resample
        return resample(self, integrator, key)
    
    def apply_roughness(self, roughness_arcsec, key):
        from .operations import apply_roughness
        return apply_roughness(self, roughness_arcsec, key)