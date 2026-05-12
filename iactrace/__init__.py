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
from .core import LazyRayBundle, RayBundle
from .io import save_camera, save_telescope
from .telescope import Telescope
from .viz import show_camera, show_telescope

__version__ = "0.7.0"

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
    "show_camera",
    "show_telescope",
    # I/O
    "save_telescope",
    "save_camera",
]
