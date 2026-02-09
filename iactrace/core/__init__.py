# Optical base classes (no dependencies, import first)
from .integrators import Integrator, MCIntegrator
from .intersections import (
    intersect_box,
    intersect_conic,
    intersect_cylinder,
    intersect_oriented_box,
    intersect_plane,
    intersect_sphere,
    intersect_triangle,
    newton_raphson_intersect,
)
from .obstructions import (
    BoxGroup,
    CylinderGroup,
    ObstructionGroup,
    OpenCylinderGroup,
    OrientedBoxGroup,
    SphereGroup,
    TriangleGroup,
)
from .optics import (
    InteractionType,
    OpticalGroupBase,
    apply_perturbation,
    fresnel_unpolarized,
    generate_perturbation_angles,
    reflect,
    refract,
    refract_slab,
)
from .render import render, render_debug, render_response_matrix, trace_rays, trace_rays_debug
from .surfaces import AsphericSurface, compute_sag_and_normal, sag, sag_raw
from .transforms import euler_to_matrix, look_at_rotation

__all__ = [
    # Optical base classes
    "InteractionType",
    "OpticalGroupBase",

    # Intersections
    "intersect_plane",
    "intersect_cylinder",
    "intersect_box",
    "intersect_sphere",
    "intersect_oriented_box",
    "intersect_triangle",
    "intersect_conic",
    "newton_raphson_intersect",

    # Surfaces
    "AsphericSurface",
    "sag_raw",
    "sag",
    "compute_sag_and_normal",

    # Integrators
    "Integrator",
    "MCIntegrator",

    # Normals / perturbations
    "generate_perturbation_angles",
    "apply_perturbation",

    # Reflection
    "reflect",

    # Refraction
    "refract",
    "refract_slab",
    "fresnel_unpolarized",

    # Transforms
    "euler_to_matrix",
    "look_at_rotation",

    # Rendering
    "render",
    "render_debug",
    "render_response_matrix",
    "trace_rays",
    "trace_rays_debug",

    # Obstructions (groups only)
    "ObstructionGroup",
    "CylinderGroup",
    "OpenCylinderGroup",
    "BoxGroup",
    "SphereGroup",
    "OrientedBoxGroup",
    "TriangleGroup",
]
