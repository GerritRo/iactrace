import jax

from .camera import (
    Camera,
    Concentrator,
    DetectionChain,
    HexagonalSensorGroup,
    OkumuraCone,
    PhotoSensor,
    SensorGroup,
    SquareSensorGroup,
    UniformQE,
    WinstonCone,
)
from .core import LazyRayBundle, RayBundle
from .io import save_camera, save_telescope
from .telescope import Telescope
from .viz import show_camera, show_sensor_chain, show_telescope

__version__ = "0.8.0"

jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "Telescope",
    "Camera",
    "RayBundle",
    "LazyRayBundle",
    "Concentrator",
    "WinstonCone",
    "OkumuraCone",
    "DetectionChain",
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
    "show_sensor_chain",
    # I/O
    "save_telescope",
    "save_camera",
]