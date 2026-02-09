import enum
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random
from jax import Array


class InteractionType(enum.Enum):
    """Type of optical interaction at a surface."""
    REFLECT = "reflect"      # Mirror - reflects rays
    REFRACT = "refract"      # Single refractive surface
    SLAB = "slab"            # Parallel-sided window (two surfaces)


class OpticalGroupBase(eqx.Module):
    """Abstract base class for all optical element groups.

    Defines the interface that the rendering engine uses to interact with
    optical elements. Concrete implementations (mirrors, lenses, slabs) are
    defined in the telescope module.

    Required attributes (to be defined by subclasses):
        positions: (N, 3) center positions
        rotations: (N, 3) Euler angles in degrees
        optical_stage: int, 0=primary, 1=secondary, etc.
    """
    positions: "Array"  # (N, 3) center positions
    rotations: "Array"  # (N, 3) Euler angles in degrees

    @property
    @abstractmethod
    def interaction(self) -> InteractionType:
        """Return the type of optical interaction for this group."""
        ...

    @abstractmethod
    def get_surface(self, element_idx):
        """Return the surface object for a specific element."""
        ...

    @abstractmethod
    def check_aperture(self, x, y, element_idx):
        """Check if points (x, y) are within aperture of specified element."""
        ...

    @abstractmethod
    def get_sampling_params(self):
        """Return structured dict with geometry parameters for sampling."""
        ...

    def __len__(self):
        """Return number of elements in group."""
        return self.positions.shape[0]

    @property
    def n_elements(self) -> int:
        """Return number of elements in group."""
        return self.positions.shape[0]


def reflect(direction, normal):
    """
    Reflect ray direction off surface with given normal.

    Implements the law of reflection: the angle of incidence equals the
    angle of reflection, with the reflected ray in the plane of incidence.

    Args:
        direction: Ray direction (3,), pointing towards surface
        normal: Surface normal (3,), pointing outward from surface

    Returns:
        reflected: Reflected direction (3,)
        cos_angle: Cosine of incident angle (positive value)
    """
    cos_angle = jnp.sum(direction * normal, axis=-1, keepdims=True)
    reflected = direction - 2.0 * cos_angle * normal
    return reflected, -cos_angle


def refract(direction, normal, n1, n2):
    """
    Refract ray direction through interface using Snell's law.

    Computes the refracted ray direction when light passes from a medium
    with refractive index n1 into a medium with refractive index n2.

    The function handles rays coming from either side of the surface by
    checking the sign of the dot product and flipping the normal if needed.

    Args:
        direction: Ray direction (3,), normalized, pointing towards surface
        normal: Surface normal (3,), normalized, pointing outward
        n1: Refractive index of incident medium (scalar)
        n2: Refractive index of transmitted medium (scalar)

    Returns:
        refracted: Refracted direction (3,), or original direction if TIR
        cos_theta_t: Cosine of refracted angle
        tir: Boolean, True if total internal reflection occurred
    """
    # Compute cosine of incident angle
    cos_i = -jnp.dot(direction, normal)

    # If ray is coming from behind the surface, flip normal
    flip = cos_i < 0
    normal = jnp.where(flip, -normal, normal)
    cos_i = jnp.where(flip, -cos_i, cos_i)

    # Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
    eta = n1 / n2
    sin2_t = eta**2 * (1.0 - cos_i**2)

    # Check for total internal reflection
    tir = sin2_t > 1.0

    # Compute refracted direction (valid only if no TIR)
    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))
    refracted = eta * direction + (eta * cos_i - cos_t) * normal

    # Normalize (should already be unit length, but numerical safety)
    refracted = refracted / jnp.linalg.norm(refracted)

    # If TIR, return reflected direction instead
    reflected = direction - 2.0 * (-cos_i) * normal
    refracted = jnp.where(tir, reflected, refracted)

    return refracted, cos_t, tir


def refract_slab(direction, normal, position, n_out, n_in, thickness):
    """
    Refract ray through a parallel-sided slab (window).

    Handles both entry and exit refractions for a slab with parallel surfaces.
    The exit surface normal is assumed to be opposite to the entry normal.
    Applies Fresnel transmission coefficients at both surfaces.

    For flat slabs, the exiting ray direction equals the entering direction
    (but with a lateral offset). For curved slabs with parallel surfaces,
    there may be slight direction change due to the curved geometry.

    Args:
        direction: Ray direction (3,), normalized
        normal: Front surface normal (3,), pointing outward (into incident medium)
        position: Entry point position (3,)
        n_out: Refractive index of ambient medium (scalar)
        n_in: Refractive index of slab material (scalar)
        thickness: Slab thickness (scalar)

    Returns:
        exit_direction: Direction after exiting slab (3,)
        exit_position: Position where ray exits slab (3,)
        transmittance: Combined Fresnel transmission coefficient for both surfaces
        valid: Boolean, True if ray successfully transmitted (no TIR)
    """
    # Entry refraction (ambient -> material)
    dir_inside, cos_t_entry, tir_entry = refract(direction, normal, n_out, n_in)

    # Compute incident angle cosine for Fresnel at entry
    cos_i_entry = jnp.abs(jnp.dot(direction, normal))

    # Fresnel transmission at entry surface
    _, T_entry = fresnel_unpolarized(cos_i_entry, cos_t_entry, n_out, n_in)

    # Propagate through slab
    # Path length inside slab (along refracted direction)
    cos_to_normal = jnp.abs(jnp.dot(dir_inside, normal))
    # Avoid division by zero for grazing angles
    cos_to_normal = jnp.maximum(cos_to_normal, 1e-10)
    path_length = thickness / cos_to_normal
    exit_position = position + path_length * dir_inside

    # Exit refraction (material -> ambient)
    # Back surface normal is opposite to front normal (parallel surfaces)
    back_normal = -normal
    exit_direction, cos_t_exit, tir_exit = refract(dir_inside, back_normal, n_in, n_out)

    # Compute incident angle cosine for Fresnel at exit (inside the slab)
    cos_i_exit = jnp.abs(jnp.dot(dir_inside, back_normal))

    # Fresnel transmission at exit surface
    _, T_exit = fresnel_unpolarized(cos_i_exit, cos_t_exit, n_in, n_out)

    # Combined validity
    valid = ~tir_entry & ~tir_exit

    # Combined Fresnel transmission (product of both surfaces)
    # If any TIR occurred, transmittance is 0
    transmittance = jnp.where(valid, T_entry * T_exit, 0.0)

    return exit_direction, exit_position, transmittance, valid


def fresnel_unpolarized(cos_theta_i, cos_theta_t, n1, n2):
    """
    Compute Fresnel reflection and transmission coefficients for unpolarized light.

    Uses the Fresnel equations to compute the fraction of light reflected (R)
    and transmitted (T) at an interface, averaged over s and p polarizations.

    Args:
        cos_theta_i: Cosine of incident angle (scalar or array)
        cos_theta_t: Cosine of transmitted angle (scalar or array)
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium

    Returns:
        R: Reflectance (fraction of light reflected), 0 <= R <= 1
        T: Transmittance (fraction of light transmitted), T = 1 - R
    """
    # s-polarization (perpendicular to plane of incidence)
    rs_num = n1 * cos_theta_i - n2 * cos_theta_t
    rs_den = n1 * cos_theta_i + n2 * cos_theta_t
    # Avoid division by zero
    rs_den = jnp.where(jnp.abs(rs_den) < 1e-10, 1e-10, rs_den)
    rs = (rs_num / rs_den)**2

    # p-polarization (parallel to plane of incidence)
    rp_num = n2 * cos_theta_i - n1 * cos_theta_t
    rp_den = n2 * cos_theta_i + n1 * cos_theta_t
    rp_den = jnp.where(jnp.abs(rp_den) < 1e-10, 1e-10, rp_den)
    rp = (rp_num / rp_den)**2

    # Average for unpolarized light
    R = 0.5 * (rs + rp)
    T = 1.0 - R

    return R, T


def generate_perturbation_angles(normals, key):
    """
    Generate random perturbation angles for surface roughness.

    Args:
        normals: Surface normals (..., 3) - used only for shape
        key: JAX random key

    Returns:
        angles: Random angles (..., 2) as (theta1, theta2) pairs
    """
    shape = normals.shape[:-1]

    key1, key2 = jax.random.split(key)
    theta1 = jax.random.normal(key1, shape)
    theta2 = jax.random.normal(key2, shape)

    return jnp.stack([theta1, theta2], axis=-1)


def apply_perturbation(normals, angles, scale):
    """
    Apply perturbation to normals using stored random angles.

    Computes tangent basis from current normals and applies the random
    angles scaled by the perturbation scale.

    Args:
        normals: Surface normals (..., 3), assumed unit length
        angles: Random angles (..., 2) as (theta1, theta2) pairs
        scale: Perturbation scale in radians (scalar)

    Returns:
        Perturbed normals (..., 3), normalized
    """
    theta1 = angles[..., 0]
    theta2 = angles[..., 1]

    # Build tangent basis, avoiding degeneracy
    ref_z = jnp.array([0., 0., 1.])
    ref_x = jnp.array([1., 0., 0.])

    dot_z = jnp.abs(jnp.sum(normals * ref_z, axis=-1, keepdims=True))
    ref = jnp.where(dot_z > 0.9, ref_x, ref_z)

    tangent1 = jnp.cross(normals, ref)
    tangent1 = tangent1 / jnp.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = jnp.cross(normals, tangent1)

    # Compute delta and apply with scale
    delta = theta1[..., None] * tangent1 + theta2[..., None] * tangent2
    perturbed = normals + scale * delta
    return perturbed / jnp.linalg.norm(perturbed, axis=-1, keepdims=True)
