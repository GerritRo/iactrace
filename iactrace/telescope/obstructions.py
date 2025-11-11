"""Obstruction geometry classes for telescope structures."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Cylinder:
    """
    Cylindrical obstruction (e.g., mast, support structure).

    Represents a finite cylinder between two points.
    """
    p1: jnp.ndarray  # Start point (3,)
    p2: jnp.ndarray  # End point (3,)
    radius: float    # Cylinder radius

    def __post_init__(self):
        """Ensure arrays are JAX arrays."""
        self.p1 = jnp.array(self.p1)
        self.p2 = jnp.array(self.p2)


@dataclass
class Box:
    """
    Oriented box obstruction (e.g., camera housing, electronics box).

    Defined by two diagonal corner points. The box extends from p1 to p2,
    with edges parallel to the coordinate axes.
    """
    p1: jnp.ndarray      # First corner (3,)
    p2: jnp.ndarray      # Opposite diagonal corner (3,)

    def __post_init__(self):
        """Ensure arrays are JAX arrays."""
        self.p1 = jnp.array(self.p1)
        self.p2 = jnp.array(self.p2)
