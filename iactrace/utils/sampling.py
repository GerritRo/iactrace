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


def sample_polygon(key, vertices, shape):
    """Sample uniformly over a convex polygon via fan triangulation.

    Args:
        key: JAX random key
        vertices: (n, 2) array of polygon vertices in order (must be convex)
        shape: batch shape

    Returns:
        points: (..., 2) array of (x, y) coordinates
    """
    # Fan triangulation from first vertex
    n = len(vertices)
    triangles = jnp.array([[vertices[0], vertices[i], vertices[i+1]]
                           for i in range(1, n-1)])  # (n-2, 3, 2)

    # Compute triangle areas
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    areas = jnp.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
                    (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])) / 2
    probs = areas / areas.sum()

    # Sample triangle index weighted by area
    key1, key2, key3, key4 = random.split(key, 4)
    tri_idx = random.choice(key1, n - 2, shape=shape, p=probs)

    # Sample uniformly within selected triangles
    u = jnp.sqrt(random.uniform(key2, shape))
    v = random.uniform(key3, shape)
    a = (1 - u)
    b = u * (1 - v)
    c = u * v

    tri_verts = triangles[tri_idx]  # (..., 3, 2)
    points = (a[..., None] * tri_verts[:, 0] +
              b[..., None] * tri_verts[:, 1] +
              c[..., None] * tri_verts[:, 2])

    return points