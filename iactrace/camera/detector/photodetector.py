from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...core.ray_bundle import RayBundle
from .surface import DetectionSurface


def _validate_qe(qe: float) -> float:
    """Validate and coerce a quantum efficiency to a plain float in ``[0, 1]``."""
    if not 0.0 <= qe <= 1.0:
        raise ValueError(f"qe must be in [0, 1], got {qe}")
    return float(qe)


def incidence_cos(directions: Array, normals: Array) -> Array:
    """Incidence cosine of each ray on a surface with the given outward normals.

    Mirrors the convention of :class:`~iactrace.core.interactions.RefractInteraction`
    (``cos_i = -dot(direction, normal)``). Helper for angle-dependent
    photodetectors, which obtain the normals from their own geometry via
    :meth:`~iactrace.camera.detector.surface.DetectionSurface.normals_at`.
    """
    return jnp.clip(-jnp.sum(directions * normals, axis=-1), 0.0, 1.0)


class PhotoDetector(eqx.Module):
    """Abstract base for photodetector response models.

    A photodetector is the terminal element of a detection chain and owns two
    things:

    * **Its surface** (:attr:`surface`): the
      :class:`~iactrace.camera.detector.surface.DetectionSurface` the chain traces
      rays onto -- by definition every photodetector has one. The base class
      provides the default (an unbounded flat detector at the chain's detector
      plane); photodetectors with a curved / apertured photocathode override the
      property.
    * **Its response** (:meth:`detect`): it receives the rays the chain has
      delivered onto that surface and weights ``values`` by its detection
      efficiency. The handover is just the :class:`RayBundle`; a photodetector
      with an angle-dependent response reads the geometry it needs from its
      own surface (e.g. :meth:`~iactrace.camera.detector.surface.DetectionSurface.normals_at`
      at the landing positions, turned into incidence cosines with
      :func:`incidence_cos`).
    """

    @property
    def surface(self) -> DetectionSurface:
        """The sensor surface, with ``vertex_z`` relative to the detector plane."""
        return DetectionSurface()

    @abstractmethod
    def detect(self, local_rays: RayBundle) -> RayBundle:
        """Weight *local_rays* by detection efficiency at the sensor surface.

        Args:
            local_rays: Rays landed on the surface, pixel-local frame (true
                directions preserved; dead / undetected rays carry ``0``).

        Returns:
            Rays with photoelectron-weighted ``values``; geometry unchanged.
        """
        raise NotImplementedError

    def outline(self) -> Array | None:
        """Optional active-area polygon ``(M, 2)`` for the diagnostic viz.

        Expressed in the pixel-local frame. The default returns ``None``
        ("not drawable"), in which case :func:`iactrace.viz.show_sensor_chain`
        falls back to the entrance-aperture footprint.
        """
        return None

    def envelope(self) -> tuple[Array, Array] | None:
        """Optional 3D envelope ``(z, rings)`` for :func:`iactrace.viz.show_sensor_chain`.

        Mirrors :meth:`~iactrace.camera.optics.concentrator.Concentrator.cross_sections`
        for the detector side: a surface of revolution / lofted wall drawn around
        the detector plane so a physical photodetector body (e.g. a PMT's glass
        front + tube) becomes visible.

        * ``z``; shape ``(K,)`` axial samples in the pixel-local frame, with
          ``z = 0`` at the photocathode (detector) plane and ``+z`` toward the
          incoming light. The viz offsets these to
          :attr:`~iactrace.camera.detection_chain.DetectionChain.detector_z`.
        * ``rings``; shape ``(K, M, 2)`` wall cross-section at each slice
          (large ``M`` ~ round).

        The default returns ``None`` ("no envelope drawn"); photodetectors with a
        physical body override it.
        """
        return None


class ConstantQE(PhotoDetector):
    """Flat scalar quantum efficiency with no spatial or angular structure.

    The simplest photodetector and the default detector response: a single
    efficiency ``qe`` applied uniformly to every ray reaching the surface
    (the inherited flat detector at the chain's detector plane). Use it for a
    measured detection efficiency you want applied as a plain scalar, or as a
    perfect (``qe = 1``) pass-through.

    Args:
        qe: Quantum efficiency in ``[0, 1]``.
    """

    qe: float = eqx.field(static=True)

    def __init__(self, qe: float = 1.0) -> None:
        self.qe = _validate_qe(qe)

    def detect(self, local_rays: RayBundle) -> RayBundle:
        return local_rays.replace(values=local_rays.values * self.qe)
