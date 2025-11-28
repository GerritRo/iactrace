import jax
import jax.numpy as jnp


def reflect(d, n):
    """
    Reflect ray direction off surface with given normal.

    Args:
        d: Ray direction (3,)
        n: Surface normal (3,)

    Returns:
        reflected: Reflected direction (3,)
        cos_angle: Cosine of incident angle (1,)
    """
    cos_angle = jnp.sum(d * n, axis=-1, keepdims=True)
    reflected = d - 2.0 * cos_angle * n
    return reflected, -cos_angle