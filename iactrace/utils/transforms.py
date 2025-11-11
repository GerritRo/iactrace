"""Transformation and rotation utilities."""

import jax.numpy as jnp


def look_at_rotation(mirror_pos, target_pos=jnp.array([0., 0., 0.]), up=jnp.array([0., 1., 0.])):
    """
    Compute rotation matrix to look from mirror_pos towards target_pos.

    Args:
        mirror_pos: Position to look from (3,)
        target_pos: Position to look at (3,)
        up: Up vector (3,)

    Returns:
        Rotation matrix (3, 3)
    """
    forward = target_pos - mirror_pos
    forward = forward / jnp.linalg.norm(forward)

    right = jnp.cross(forward, up)
    right = right / jnp.linalg.norm(right)

    up_corrected = jnp.cross(right, forward)

    return jnp.column_stack([right, up_corrected, forward])
