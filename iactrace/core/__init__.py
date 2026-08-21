from .apertures import Aperture, DiskAperture, PolygonAperture
from .bsdf import BSDF, DoubleGaussianBSDF, GaussianBSDF
from .interactions import (
    Interaction,
    InteractionType,
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
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
from .ray_bundle import DEFAULT_WAVELENGTH, LazyRayBundle, RayBundle
from .refractive_index import (
    ConstantIndex,
    RefractiveIndex,
    SellmeierIndex,
    TabulatedIndex,
    as_refractive_index,
)
from .render import render_optics, trace_optics
from .responses import (
    ConstantResponse,
    ResponseCurve,
    TabulatedResponse,
)
from .spectrum import ConstantSpectrum, Spectrum, TabulatedSpectrum
from .surfaces import (
    AsphericSurfaceGroup,
    FreeformSurfaceGroup,
    SumSurfaceGroup,
    SurfaceGroup,
    ZernikeSurfaceGroup,
    bicubic_interp,
    zernike_terms,
)
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
    # Response curves: R/T/QE(angle, wavelength)
    "ResponseCurve",
    "ConstantResponse",
    "TabulatedResponse",
    # Refractive index n(wavelength)
    "RefractiveIndex",
    "as_refractive_index",
    "ConstantIndex",
    "TabulatedIndex",
    "SellmeierIndex",
    # Source spectrum
    "Spectrum",
    "ConstantSpectrum",
    "TabulatedSpectrum",
    # Surfaces
    "SurfaceGroup",
    "AsphericSurfaceGroup",
    "ZernikeSurfaceGroup",
    "SumSurfaceGroup",
    "FreeformSurfaceGroup",
    "zernike_terms",
    "bicubic_interp",
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
    "trace_optics",
    # Obstructions
    "ObstructionGroup",
    "CylinderGroup",
    "OpenCylinderGroup",
    "BoxGroup",
    "SphereGroup",
    "OrientedBoxGroup",
    "TriangleGroup",
]
