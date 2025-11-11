"""
IACTrace: JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes.

This package provides efficient, differentiable simulation of IACT optical systems.
"""

from .telescope.telescope import Telescope
from .telescope.obstructions import Cylinder, Box
from .samplers.mirror_out import MCIntegrator, GridIntegrator
from .sensors.square import SquareSensor
from .sensors.hexagonal import HexagonalSensor
from .viz.plotting import hexshow, plot_telescope_geometry, plot_focal_plane

__version__ = "0.1.0"

__all__ = [
    'Telescope',
    'Cylinder',
    'Box',
    'MCIntegrator',
    'GridIntegrator',
    'SquareSensor',
    'HexagonalSensor',
    'hexshow',
    'plot_telescope_geometry',
    'plot_focal_plane',
]
