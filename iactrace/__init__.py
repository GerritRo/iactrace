import jax

from .camera import (
    PMT,
    Camera,
    Concentrator,
    ConstantQE,
    DetectionChain,
    HexagonalSensorGroup,
    OkumuraCone,
    PhotoDetector,
    SensorGroup,
    SquareSensorGroup,
    WinstonCone,
)
from .core import LazyRayBundle, RayBundle
from .io import save_camera, save_telescope
from .telescope import Telescope
from .viz import show_camera, show_sensor_chain, show_telescope

__version__ = "0.9.0"

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
    # PhotoDetector
    "PhotoDetector",
    "ConstantQE",
    "PMT",
    # Visualization
    "show_camera",
    "show_telescope",
    "show_sensor_chain",
    # I/O
    "save_telescope",
    "save_camera",
]
