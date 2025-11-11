"""Reflection and surface roughness functions."""

import jax
import jax.numpy as jnp


def reflect(d, n):
    """
    Reflect ray direction off surface with given normal.

    Args:
        d: Ray direction (..., 3)
        n: Surface normal (..., 3)

    Returns:
        reflected: Reflected direction (..., 3)
        cos_angle: Cosine of incident angle (..., 1)
    """
    cos_angle = jnp.sum(d * n, axis=-1, keepdims=True)
    reflected = d - 2.0 * cos_angle * n
    return reflected, -cos_angle


def roughen_normals(n, roughness, key):
    """
    Perturb surface normals to simulate roughness (microfacet model).

    Args:
        n: Surface normals (M, N, 3)
        roughness: RMS roughness in arcseconds (scalar)
        key: JAX random key

    Returns:
        Perturbed normals (M, N, 3)
    """
    M, N, _ = n.shape
    roughness_rad = roughness * jnp.pi / (180.0 * 3600.0)

    # Generate random tangent perturbations
    key1, key2 = jax.random.split(key)
    perturb1 = jax.random.normal(key1, (M, N)) * roughness_rad
    perturb2 = jax.random.normal(key2, (M, N)) * roughness_rad

    # Construct local basis
    ref = jnp.array([0., 0., 1.])
    tangent1 = jnp.cross(n, ref[None, None, :])
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(n, tangent1)

    # Perturb and renormalize
    n_perturbed = n + perturb1[..., None] * tangent1 + perturb2[..., None] * tangent2
    return n_perturbed / jnp.linalg.norm(n_perturbed, axis=-1, keepdims=True)
