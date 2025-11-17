import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Cylinder:
    """
    Cylindrical obstruction
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
    Oriented box obstruction
    """
    p1: jnp.ndarray      # First corner (3,)
    p2: jnp.ndarray      # Opposite diagonal corner (3,)

    def __post_init__(self):
        """Ensure arrays are JAX arrays."""
        self.p1 = jnp.array(self.p1)
        self.p2 = jnp.array(self.p2)
