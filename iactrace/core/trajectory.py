from __future__ import annotations

import equinox as eqx
import numpy as np
from jax import Array


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
