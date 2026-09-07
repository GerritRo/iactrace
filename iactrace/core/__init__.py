from .apertures import Aperture, DiskAperture, PolygonAperture
from .bsdf import BSDF, DoubleGaussianBSDF, GaussianBSDF
from .interactions import (
    Interaction,
    InteractionType,
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
    reflect,
    refract,
    refract_slab,
)
from .intersections import (
    intersect_box,
    intersect_conic,
    intersect_cylinder,
    intersect_open_cylinder,
    intersect_oriented_box,
    intersect_plane,
    intersect_sphere,
    intersect_triangle,
    is_hit,
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
from .optics import OpticalElementGroup
from .ray_bundle import DEFAULT_WAVELENGTH, RayBundle
from .refractive_index import (
    ConstantIndex,
    RefractiveIndex,
    SellmeierIndex,
    TabulatedIndex,
    as_refractive_index,
)
from .render import (
    LazyRayBundle,
    apply_final_leg_shadow,
    final_leg_points,
    handoff_to_frame,
    render_optics,
    render_optics_accumulate,
    trace_optics,
)
from .responses import (
    ConstantResponse,
    ResponseCurve,
    TabulatedResponse,
    fresnel_unpolarized,
)
from .sampling import sample_annulus, sample_polygon
from .spectrum import ConstantSpectrum, Spectrum, TabulatedSpectrum, as_spectrum
from .surfaces import (
    N_ZERNIKE,
    AsphericSurfaceGroup,
    FreeformSurfaceGroup,
    SumSurfaceGroup,
    SurfaceGroup,
    ZernikeSurfaceGroup,
    bicubic_interp,
    compute_sag_and_normal,
    sag,
    sag_raw,
    zernike_terms,
)
from .tolerances import dir_tol, len_rel
from .trajectory import TraceResult, Trajectory
from .transforms import euler_to_matrix

__all__ = [
    # Optical element group
    "InteractionType",
    "OpticalElementGroup",
    # Aperture modules
    "Aperture",
    "DiskAperture",
    "PolygonAperture",
    # Interaction modules
    "Interaction",
    "ReflectInteraction",
    "RefractInteraction",
    "SlabInteraction",
    # Interaction physics
    "reflect",
    "refract",
    "refract_slab",
    "fresnel_unpolarized",
    # Response curves
    "ResponseCurve",
    "ConstantResponse",
    "TabulatedResponse",
    # Refractive index
    "RefractiveIndex",
    "as_refractive_index",
    "ConstantIndex",
    "TabulatedIndex",
    "SellmeierIndex",
    # Source spectrum
    "Spectrum",
    "ConstantSpectrum",
    "TabulatedSpectrum",
    "as_spectrum",
    # Surfaces
    "SurfaceGroup",
    "AsphericSurfaceGroup",
    "ZernikeSurfaceGroup",
    "SumSurfaceGroup",
    "FreeformSurfaceGroup",
    "zernike_terms",
    "bicubic_interp",
    "sag",
    "sag_raw",
    "compute_sag_and_normal",
    "N_ZERNIKE",
    # BSDF
    "BSDF",
    "GaussianBSDF",
    "DoubleGaussianBSDF",
    # Transforms
    "euler_to_matrix",
    # Ray bundle
    "RayBundle",
    "LazyRayBundle",
    "DEFAULT_WAVELENGTH",
    # Trajectory
    "Trajectory",
    "TraceResult",
    # Render engine
    "render_optics",
    "render_optics_accumulate",
    "trace_optics",
    # Handoff from the optics to a local (camera) frame
    "apply_final_leg_shadow",
    "final_leg_points",
    "handoff_to_frame",
    # Obstructions
    "ObstructionGroup",
    "CylinderGroup",
    "OpenCylinderGroup",
    "BoxGroup",
    "SphereGroup",
    "OrientedBoxGroup",
    "TriangleGroup",
    # Ray-primitive intersection kernels
    "intersect_plane",
    "intersect_sphere",
    "intersect_cylinder",
    "intersect_open_cylinder",
    "intersect_box",
    "intersect_oriented_box",
    "intersect_triangle",
    "intersect_conic",
    "newton_raphson_intersect",
    "is_hit",
    # Aperture sampling
    "sample_annulus",
    "sample_polygon",
    # Numerical tolerances
    "dir_tol",
    "len_rel",
]
