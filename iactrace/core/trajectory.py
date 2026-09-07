from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import numpy as np
from jax import Array

from .ray_bundle import RayBundle


class Trajectory(eqx.Module):
    """Per-step 3-D positions of a bundle of rays, recorded through a trace.

    Attributes
    ----------
    points
        (steps + 1, n_rays, 3) per-step ray positions.
    """

    points: Array

    def __array__(self, dtype=None) -> np.ndarray:
        """Expose the raw (steps + 1, n_rays, 3) points to np.asarray."""
        return np.asarray(self.points, dtype=dtype)


class TraceResult(NamedTuple):
    """A traced RayBundle.

    Attributes
    ----------
    rays
        The traced bundle, in whatever frame the tracer works in.
    trajectory
        Per-step positions through the trace, or None when the
        trace was run without record_trajectory.
    """

    rays: RayBundle
    trajectory: Trajectory | None = None
