import jax

from .camera import (
    Camera,
    Concentrator,
    HexagonalSensorGroup,
    PhotoSensor,
    SensorGroup,
    SquareSensorGroup,
    UniformQE,
)
from .core import RayBundle, LazyRayBundle
from .io import save_camera, save_telescope
from .telescope import Telescope
from .viz import hexshow, show_telescope, squareshow

__version__ = "0.6.1"

jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "Telescope",
    "Camera",
    "RayBundle",
    "LazyRayBundle",
    "Concentrator",
    # Sensors
    "SensorGroup",
    "SquareSensorGroup",
    "HexagonalSensorGroup",
    # PhotoSensor
    "PhotoSensor",
    "UniformQE",
    # Visualization
    "hexshow",
    "squareshow",
    "show_telescope",
    # I/O
    "save_telescope",
    "save_camera",
]