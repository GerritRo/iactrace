from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ..core.ray_bundle import RayBundle


class Concentrator(eqx.Module):
    """Abstract base for per-pixel light concentrators (cones / lightguides).

    It funnels light from its entrance aperture (``z = 0``) to its exit
    aperture (``z = -length``) in its local space.

    Concrete physics (Winston/CPC walls, reflection losses, time dispersion)
    are implemented by subclasses. It is agnostic to implementation specifics.
    """

    length: eqx.AbstractVar[float]

    @property
    def index(self) -> float:
        """Refractive index of the medium the concentrator is filled with.

        ``1.0`` for hollow / air-filled light guides such as Winston cones.
        Solid dielectric concentrators override this; :meth:`apply` must weight
        the internal geometric path by it when accumulating optical path length
        (``OPL += index * geometric_length``).
        """
        return 1.0

    @abstractmethod
    def apply(self, local_rays: RayBundle) -> RayBundle:
        """Transport *local_rays* from the entrance to the exit aperture.

        Rays enter in the ``z = 0`` plane (travelling toward ``-z``) and leave in
        the ``z = -length`` plane. Implementations:

        * attenuate ``values`` by reflection / collection losses,
        * update ``origins`` / ``directions`` to the exit-ray state,
        * add the **optical** path travelled, and
        * set ``n`` to the refractive index of the medium **at the exit
          aperture**.

        Args:
            local_rays: Rays at the entrance aperture, pixel-local frame.

        Returns:
            Rays at the exit aperture, same frame.
        """
        raise NotImplementedError

    def cross_sections(self) -> tuple[Array, Array] | None:
        """Optional geometry for :func:`iactrace.viz.show_sensor_chain`.

        Returns ``(z, rings)`` or ``None``:

        * ``z``; shape ``(K,)`` axial samples, ``z[0] = 0`` (entrance) ..
          ``z[-1] = -length`` (exit).
        * ``rings``; shape ``(K, M, 2)``: the ``M``-gon wall cross-section at
          each slice in the pixel-local frame (``M = 6`` hex, ``M = 4`` square,
          large ``M`` ~ round). ``rings[0]`` is the entrance aperture,
          ``rings[-1]`` the exit aperture.

        The default returns ``None`` ("not drawable"); concrete concentrators
        override it once they know their profile.
        """
        return None
