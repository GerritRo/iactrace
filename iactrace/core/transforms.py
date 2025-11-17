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


def euler_to_matrix(tip_tilt_rotation):
    """
    Convert Euler angles (degrees) to rotation matrix.
    
    Args:
        tip_tilt_rotation: List of all 3 transformations(3,)
    
    Returns:
        Rotation matrix (3, 3)
    """
    tip, tilt, rotation = tip_tilt_rotation[0], tip_tilt_rotation[1], tip_tilt_rotation[2]
    
    # Convert to radians
    rx, ry, rz = jnp.radians(jnp.array([tip, tilt, rotation]))
    
    # Rotation matrices
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(rx), -jnp.sin(rx)],
        [0, jnp.sin(rx), jnp.cos(rx)]
    ])
    
    Ry = jnp.array([
        [jnp.cos(ry), 0, jnp.sin(ry)],
        [0, 1, 0],
        [-jnp.sin(ry), 0, jnp.cos(ry)]
    ])
    
    Rz = jnp.array([
        [jnp.cos(rz), -jnp.sin(rz), 0],
        [jnp.sin(rz), jnp.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Apply: Rz * Ry * Rx (extrinsic order)
    return Rz @ Ry @ Rx