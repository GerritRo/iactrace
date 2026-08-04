from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import numpy as np
from jax import Array

from .ray_bundle import RayBundle


class Trajectory(eqx.Module):
    """Per-step 3-D positions of a bundle of rays, recorded through a trace.

    Attributes:
        points: ``(steps + 1, n_rays, 3)`` per-step ray positions.
    """

    points: Array

    @property
    def n_steps(self) -> int:
        """Number of interaction steps -- ``points.shape[0] - 1``."""
        return self.points.shape[0] - 1

    @property
    def n_rays(self) -> int:
        """Number of rays -- ``points.shape[1]``."""
        return self.points.shape[1]

    def __array__(self, dtype=None) -> np.ndarray:
        """Expose the raw ``(steps + 1, n_rays, 3)`` points to ``np.asarray``."""
        return np.asarray(self.points, dtype=dtype)


class TraceResult(NamedTuple):
    """A traced :class:`~iactrace.core.ray_bundle.RayBundle` and the path it took.

    Returned by every tracer that can optionally record a path
    (:func:`~iactrace.core.render.trace_optics`,
    :meth:`~iactrace.camera.DetectionChain.propagate`,
    :meth:`~iactrace.camera.optics.concentrator.Concentrator.trace_to_surface`).

    Attributes:
        rays: The traced bundle, in whatever frame the tracer works in.
        trajectory: Per-step positions through the trace, or ``None`` when the
            trace was run without ``record_trajectory`` (and for a tracer that
            cannot report a path at all -- callers fall back to a straight
            segment).
    """

    rays: RayBundle
    trajectory: Trajectory | None = None
