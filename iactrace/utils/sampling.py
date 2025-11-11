"""Sampling utilities for ray generation."""

import jax
import jax.numpy as jnp
from jax import random


def sample_disk(key, shape):
    """
    Generate uniform random samples within a unit disk.

    Args:
        key: JAX random key
        shape: Shape of samples to generate

    Returns:
        2D points in unit disk (..., 2)
    """
    key1, key2 = random.split(key)

    r = jnp.sqrt(random.uniform(key1, shape))
    theta = random.uniform(key2, shape) * 2 * jnp.pi

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)

    return jnp.stack([x, y], axis=-1)


def sample_grid(n_samples):
    """
    Generate uniform grid samples within a unit disk.

    Args:
        n_samples: Number of samples along one dimension

    Returns:
        2D points in unit disk (N, 2) where N <= n_samples^2
    """
    x = jnp.linspace(-1, 1, n_samples)
    y = jnp.linspace(-1, 1, n_samples)
    xx, yy = jnp.meshgrid(x, y)
    points = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)

    # Filter to unit disk
    r = jnp.linalg.norm(points, axis=-1)
    points = points[r <= 1.0]

    return points
