import jax

from .core import Integrator, MCIntegrator
from .io import build_telescope, load_telescope, save_telescope
from .sensors import (
    HexagonalSensorGroup,
    SensorGroup,
    SquareSensorGroup,
    StraightThroughHexagonalSensorGroup,
    StraightThroughSquareSensorGroup,
)
from .telescope import Telescope
from .viz import hexshow, show_telescope, squareshow

__version__ = "0.6.1"

jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "Telescope",
    "Integrator",
    "MCIntegrator",
    # Sensors
    "SensorGroup",
    "SquareSensorGroup",
    "StraightThroughSquareSensorGroup",
    "HexagonalSensorGroup",
    "StraightThroughHexagonalSensorGroup",
    # Visualization
    "hexshow",
    "squareshow",
    "show_telescope",
    # I/O
    "load_telescope",
    "build_telescope",
    "save_telescope",
]
