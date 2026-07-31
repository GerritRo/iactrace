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
from .core import LazyRayBundle, RayBundle, Trajectory
from .io import save_camera, save_telescope
from .telescope import Telescope
from .viz import (
    add_trajectories,
    show_camera,
    show_image,
    show_sensor_chain,
    show_telescope,
)

__version__ = "0.8.0"

jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "Telescope",
    "Camera",
    "RayBundle",
    "LazyRayBundle",
    "Trajectory",
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
    "show_image",
    "show_telescope",
    "show_camera",
    "show_sensor_chain",
    "add_trajectories",
    # I/O
    "save_telescope",
    "save_camera",
]