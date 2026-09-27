from __future__ import annotations

import jax
import jax.numpy as jnp


def euler_to_matrix(tip_tilt_rotation):
    """(3, 3) rotation from Euler angles (rx, ry, rz) in degrees.

    Applied extrinsically, x then y then z: R = Rz(rz) @ Ry(ry) @ Rx(rx).
    """
    rx, ry, rz = jnp.radians(jnp.asarray(tip_tilt_rotation))
    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(rx), -jnp.sin(rx)], [0, jnp.sin(rx), jnp.cos(rx)]])
    Ry = jnp.array([[jnp.cos(ry), 0, jnp.sin(ry)], [0, 1, 0], [-jnp.sin(ry), 0, jnp.cos(ry)]])
    Rz = jnp.array([[jnp.cos(rz), -jnp.sin(rz), 0], [jnp.sin(rz), jnp.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform_to_world(aperture_samples, surface, aperture_data, positions, rotations, area_fn):
    """Lift sampled element geometry from local coordinates into the world frame.

    Shared by every optical group type (mirrors, lenses, slabs). Evaluates the
    surface at the (N, M, 2) aperture_samples, then applies each
    element's rotation and translation. surface is vmapped over axis 0, so
    each iteration sees a single element with scalar parameters.

    Returns (points_world, normals_world, weights), shaped (N, M, 3),
    (N, M, 3) and (N, M, 1); weights are the geometry integration
    weights, cos(angle to z) / area * M.
    """
    # The unbound method for the concrete surface type: after vmapping,
    # surf_single has scalar curvature, (2,) offset, etc.
    sag_normal_method = type(surface).compute_sag_and_normal_at

    def compute_and_transform_single(xy, surf_single, ap_data, position, rotation):
        x, y = xy[..., 0], xy[..., 1]
        points, normals = jax.vmap(lambda xi, yi: sag_normal_method(surf_single, xi, yi))(x, y)

        cos_z = jnp.sum(normals * jnp.array([0.0, 0.0, 1.0]), axis=-1, keepdims=True)
        n_samples = xy.shape[0]
        area = area_fn(ap_data)
        weights = cos_z / area * n_samples

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
