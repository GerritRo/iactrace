import jax
import jax.numpy as jnp


def euler_to_matrix(tip_tilt_rotation):
    """
    Convert Euler angles (degrees) to rotation matrix.

    The angles are applied **extrinsically, x then y then z**:

        R = Rz(rz) @ Ry(ry) @ Rx(rx)

    Args:
        tip_tilt_rotation: Euler angles ``(rx, ry, rz)`` in degrees

    Returns:
        Rotation matrix (3, 3)
    """
    tip, tilt, rotation = tip_tilt_rotation[0], tip_tilt_rotation[1], tip_tilt_rotation[2]

    # Convert to radians
    rx, ry, rz = jnp.radians(jnp.array([tip, tilt, rotation]))

    # Rotation matrices
    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(rx), -jnp.sin(rx)], [0, jnp.sin(rx), jnp.cos(rx)]])

    Ry = jnp.array([[jnp.cos(ry), 0, jnp.sin(ry)], [0, 1, 0], [-jnp.sin(ry), 0, jnp.cos(ry)]])

    Rz = jnp.array([[jnp.cos(rz), -jnp.sin(rz), 0], [jnp.sin(rz), jnp.cos(rz), 0], [0, 0, 1]])

    # Apply: Rz * Ry * Rx (extrinsic order)
    return Rz @ Ry @ Rx


def matrix_to_euler(rotation_matrix):
    """Convert a 3x3 rotation matrix to Euler angles (degrees).

    The inverse of :func:`~iactrace.core.transforms.euler_to_matrix`, i.e. it
    decomposes ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``.

    Args:
        Rotation matrix (3, 3)
        
    Returns:
        tip_tilt_rotation: Euler angles ``(rx, ry, rz)`` in degrees
    """
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        rx = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        ry = np.arctan2(-rotation_matrix[2, 0], sy)
        rz = 0.0
    return [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))]


def transform_to_world(aperture_samples, surface, aperture_data, positions, rotations, area_fn):
    """Transform optical element geometry from local to world coordinates.

    Shared by all optical group types (mirrors, lenses, slabs). Computes 3D
    surface geometry from aperture samples and surface parameters, then applies
    the per-element rotation and translation.

    The surface module is vmapped over axis 0; each iteration gets a
    single-element surface with scalar parameters.

    Args:
        aperture_samples: 2D sample positions on aperture (N, M, 2)
        surface: Surface module (e.g. AsphericSurfaceGroup) with (N, ...) arrays.
                 Must have a compute_sag_and_normal_at(self, x, y) method.
        aperture_data: Aperture-specific data passed to area_fn (e.g. radii, vertices)
        positions: Element positions (N, 3)
        rotations: Element rotations (N, 3) Euler angles in degrees
        area_fn: Function to compute aperture area from a single element's aperture_data

    Returns:
        Tuple of (points_world, normals_world, weights):
            - points_world: (N, M, 3) world-space sample points
            - normals_world: (N, M, 3) world-space surface normals
            - weights: (N, M, 1) geometry integration weights
    """
    # Get the unbound method for the concrete surface type.
    # After vmapping, surf_single has scalar curvature, (2,) offset, etc.
    sag_normal_method = type(surface).compute_sag_and_normal_at

    def compute_and_transform_single(xy, surf_single, ap_data, position, rotation):
        x, y = xy[..., 0], xy[..., 1]
        points, normals = jax.vmap(lambda xi, yi: sag_normal_method(surf_single, xi, yi))(x, y)

        # Compute weights: cos(angle to z-axis) / area * n_samples
        cos_z = jnp.sum(normals * jnp.array([0.0, 0.0, 1.0]), axis=-1, keepdims=True)
        n_samples = xy.shape[0]
        area = area_fn(ap_data)
        weights = cos_z / area * n_samples

        # Transform to world coordinates
        rot = euler_to_matrix(rotation)
        points_world = jnp.einsum("ij,nj->ni", rot, points) + position
        normals_world = jnp.einsum("ij,nj->ni", rot, normals)

        return points_world, normals_world, weights

    return jax.vmap(compute_and_transform_single)(
        aperture_samples,
        surface,
        aperture_data,
        positions,
        rotations,
    )
