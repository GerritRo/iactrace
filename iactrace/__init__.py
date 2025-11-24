"""
IACTrace: JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes.

This package provides efficient, differentiable simulation of IACT optical systems.
"""

from .telescope.telescope import Telescope
from .telescope.obstructions import Cylinder, Box
from .telescope.integrators import MCIntegrator
from .sensors.square import SquareSensor
from .sensors.hexagonal import HexagonalSensor
from .viz.plotting import hexshow, squareshow

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
    'squareshow',
]
