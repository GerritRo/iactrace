from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array


class PhotoSensor(eqx.Module):
    """Abstract base for photosensor response models."""

    @abstractmethod
    def apply(self, values: Array) -> Array:
        """Scale *values* by the quantum efficiency.

        Args:
            values: Photon-equivalent values (n_rays,).

        Returns:
            Photoelectron-equivalent values (n_rays,).
        """
        raise NotImplementedError


class UniformQE(PhotoSensor):
    """Scalar quantum efficiency."""

    qe: float = eqx.field(static=True)

    def __init__(self, qe: float = 1.0) -> None:
        self.qe = float(qe)

    def apply(self, values: Array) -> Array:
        return values * self.qe
