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


def roughen_normals(n, roughness, key):
    """
    Perturb surface normals to simulate roughness.

    Args:
        n: Surface normals (M, N, 3)
        roughness: RMS roughness in arcseconds
        key: JAX random key

    Returns:
        Perturbed normals (M, N, 3)
    """
    shape = n.shape[:-1]
    roughness_rad = roughness * jnp.pi / (180.0 * 3600.0)

    key1, key2 = jax.random.split(key)
    perturb1 = jax.random.normal(key1, shape) * roughness_rad
    perturb2 = jax.random.normal(key2, shape) * roughness_rad

    # Use reference that avoids degeneracy
    ref_z = jnp.array([0., 0., 1.])
    ref_x = jnp.array([1., 0., 0.])
    
    # Check alignment with z-axis
    dot_z = jnp.abs(jnp.sum(n * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)
    
    tangent1 = jnp.cross(n, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(n, tangent1)

    n_perturbed = n + perturb1[..., None] * tangent1 + perturb2[..., None] * tangent2
    return n_perturbed / jnp.linalg.norm(n_perturbed, axis=-1, keepdims=True)