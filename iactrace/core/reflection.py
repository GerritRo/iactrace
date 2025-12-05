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

def perturb_normals(normals, sigma_rad, key):
    """
    Perturb normals by random angles.
    
    Args:
        normals: Surface normals (..., 3), assumed unit length
        sigma_rad: RMS perturbation angle in radians
        key: JAX random key
    
    Returns:
        Perturbed unit normals (..., 3)
    """
    shape = normals.shape[:-1]
    
    key1, key2 = jax.random.split(key)
    theta1 = jax.random.normal(key1, shape) * sigma_rad
    theta2 = jax.random.normal(key2, shape) * sigma_rad
    
    # Build tangent basis, avoiding degeneracy
    ref_z = jnp.array([0., 0., 1.])
    ref_x = jnp.array([1., 0., 0.])
    
    dot_z = jnp.abs(jnp.sum(normals * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)
    
    tangent1 = jnp.cross(normals, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(normals, tangent1)
    
    perturbed = normals + theta1[..., None] * tangent1 + theta2[..., None] * tangent2
    return perturbed / jnp.linalg.norm(perturbed, axis=-1, keepdims=True)