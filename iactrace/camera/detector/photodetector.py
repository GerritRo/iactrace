from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...core.ray_bundle import RayBundle
from ...core.responses import ResponseCurve, TabulatedResponse
from .surface import DetectionSurface


def _validate_qe(qe: float) -> float:
    """Validate and coerce a quantum efficiency to a plain float in [0, 1]."""
    if not 0.0 <= qe <= 1.0:
        raise ValueError(f"qe must be in [0, 1], got {qe}")
    return float(qe)


def incidence_cos(directions: Array, normals: Array) -> Array:
    """Incidence cosine of each ray on a surface with the given outward normals."""
    return jnp.clip(-jnp.sum(directions * normals, axis=-1), 0.0, 1.0)


class PhotoDetector(eqx.Module):
    """Abstract base for photodetector response models.

    A photodetector is the terminal element of a detection chain and owns two
    things:

    - Its surface (surface): the
      DetectionSurface the chain traces
      rays onto -- by definition every photodetector has one. The base class
      provides the default (an unbounded flat detector at the chain's detector
      plane); photodetectors with a curved / apertured photocathode override the
      property.
    - Its response (detect): it receives the rays the chain has
      delivered onto that surface and weights values by its detection
      efficiency. The handover is just the RayBundle; a photodetector
      with an angle-dependent response reads the geometry it needs from its
      own surface (e.g. normals_at
      at the landing positions, turned into incidence cosines with
      incidence_cos).
    """

    @property
    def surface(self) -> DetectionSurface:
        """The sensor surface, with vertex_z relative to the detector plane."""
        return DetectionSurface()

    @abstractmethod
    def detect(self, local_rays: RayBundle) -> RayBundle:
        """Weight local_rays by detection efficiency at the sensor surface.

        Parameters
        ----------
        local_rays
            Rays landed on the surface, pixel-local frame (true
            directions preserved; dead / undetected rays carry 0).

        Returns
        -------
        Rays with photoelectron-weighted values; geometry unchanged.
        """
        raise NotImplementedError

    def outline(self) -> Array | None:
        """Optional active-area polygon (M, 2) for the diagnostic viz.

        Expressed in the pixel-local frame. The default returns None
        ("not drawable"), in which case iactrace.viz.show_sensor_chain
        falls back to the entrance-aperture footprint.
        """
        return None

    def envelope(self) -> tuple[Array, Array] | None:
        """Optional 3D envelope (z, rings) for iactrace.viz.show_sensor_chain.

        Mirrors cross_sections
        for the detector side: a surface of revolution / lofted wall drawn around
        the detector plane so a physical photodetector body (e.g. a PMT's glass
        front + tube) becomes visible.

        - z; shape (K,) axial samples in the pixel-local frame, with
          z = 0 at the photocathode (detector) plane and +z toward the
          incoming light. The viz offsets these to
          detector_z.
        - rings; shape (K, M, 2) wall cross-section at each slice
          (large M ~ round).

        The default returns None ("no envelope drawn"); photodetectors with a
        physical body override it.
        """
        return None


def apply_qe(
    rays: RayBundle,
    qe: float,
    qe_curve: ResponseCurve | None,
    cos_theta_i: Array | None = None,
) -> Array:
    """Photoelectron-weighted values for rays: qe * qe_curve(theta, lambda).

    Parameters
    ----------
    rays
        Rays landed on the sensor surface.
    qe
        Bulk quantum efficiency in [0, 1].
    qe_curve
        Optional response curve, or None for a flat response.
    cos_theta_i : array, shape (n_rays,)
        Incidence cosines at the surface.
        None (the default) evaluates the curve at normal incidence,
        which is exact for the usual wavelength-only QE(lambda).
    """
    values = rays.values * qe
    if qe_curve is None:
        return values
    n = rays.wavelength.shape[0]
    cos_i = jnp.ones(n) if cos_theta_i is None else cos_theta_i
    idx = jnp.zeros(n, dtype=jnp.int32)  # one photodetector = one element
    return values * qe_curve(cos_i, idx, rays.wavelength)


class ConstantQE(PhotoDetector):
    """Flat scalar quantum efficiency with no spatial, angular or spectral structure.

    Parameters
    ----------
    qe
        Quantum efficiency in [0, 1].
    """

    qe: float = eqx.field(static=True)

    def __init__(self, qe: float = 1.0) -> None:
        self.qe = _validate_qe(qe)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        return local_rays.replace(values=apply_qe(local_rays, self.qe, None))


class TabulatedQE(PhotoDetector):
    """Quantum efficiency from a ResponseCurve.

    The same qe / qe_curve pair a PMT
    uses, without the body or entrance window: the bulk qe scaled per
    ray by qe_curve. Build the usual QE(lambda) case with
    from_table.

    Attributes
    ----------
    qe
        Bulk quantum efficiency in [0, 1], multiplying the curve.
    qe_curve
        The QE(theta, lambda) response curve.
    """

    qe: float = eqx.field(static=True)
    qe_curve: ResponseCurve

    def __init__(self, qe_curve: ResponseCurve, qe: float = 1.0) -> None:
        self.qe_curve = qe_curve
        self.qe = _validate_qe(qe)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        cos_theta_i = incidence_cos(
            local_rays.directions, self.surface.normals_at(local_rays.origins)
        )
        return local_rays.replace(values=apply_qe(local_rays, self.qe, self.qe_curve, cos_theta_i))

    @classmethod
    def from_table(cls, wavelengths, qe, bulk_qe: float = 1.0) -> TabulatedQE:
        """Build from measured QE(lambda) samples.

        Parameters
        ----------
        wavelengths : array, shape (K,)
            Sample wavelengths (sorted internally).
        qe : array, shape (K,)
            Detection efficiency in [0, 1] aligned with wavelengths.
        bulk_qe
            Optional bulk multiplier applied on top of the curve.
        """
        return cls(
            qe_curve=TabulatedResponse.from_wavelengths(wavelengths, qe, n_elements=1),
            qe=bulk_qe,
        )
