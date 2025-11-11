"""Surface generation functions for mirror facets."""

import jax.numpy as jnp


def spherical_surface(radius, points):
    """
    Generate 3D points and normals on a spherical surface.

    Args:
        radius: Sphere radius (scalar)
        points: 2D coordinates on mirror (..., 2)

    Returns:
        positions: 3D positions on sphere (..., 3)
        normals: Surface normals (..., 3)
    """
    Z = radius - jnp.sqrt(radius**2 - (points[..., 0]**2 + points[..., 1]**2))
    N = jnp.stack([-points[..., 0], -points[..., 1], radius - Z], axis=-1)
    positions = jnp.stack([points[..., 0], points[..., 1], Z], axis=-1)

    return positions, N / radius


def parabolic_surface(focal_length, points):
    """
    Generate 3D points and normals on a parabolic surface.

    Args:
        focal_length: Focal length of parabola (scalar)
        points: 2D coordinates on mirror (..., 2)

    Returns:
        positions: 3D positions on parabola (..., 3)
        normals: Surface normals (..., 3)
    """
    a = 1.0 / (4.0 * focal_length)
    Z = a * (points[..., 0]**2 + points[..., 1]**2)
    ones = jnp.ones_like(Z)
    N = jnp.stack([-2*a*points[..., 0], -2*a*points[..., 1], ones], axis=-1)
    norm = jnp.sqrt(4*a*Z + 1)
    positions = jnp.stack([points[..., 0], points[..., 1], Z], axis=-1)

    return positions, N / norm[..., None]
