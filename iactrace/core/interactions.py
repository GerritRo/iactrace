import enum
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


# Interaction type enum

class InteractionType(enum.Enum):
    """Type of optical interaction at a surface."""
    REFLECT = "reflect"
    REFRACT = "refract"
    SLAB = "slab"


# Interaction base class

class Interaction(eqx.Module):
    """Abstract base for optical interaction modules."""

    @property
    @abstractmethod
    def interaction_type(self) -> InteractionType: ...

    @abstractmethod
    def apply(self, directions, normals, points, element_idx): ...


# Core physics functions

def reflect(direction, normal):
    """Reflect ray direction off surface with given normal.

    Args:
        direction: Ray direction (3,), pointing towards surface.
        normal: Surface normal (3,), pointing outward from surface.

    Returns:
        reflected: Reflected direction (3,).
        cos_angle: Cosine of incident angle (positive value).
    """
    cos_angle = jnp.sum(direction * normal, axis=-1, keepdims=True)
    reflected = direction - 2.0 * cos_angle * normal
    return reflected, -cos_angle


def refract(direction, normal, n1, n2):
    """Refract ray direction through interface using Snell's law.

    Handles rays from either side of the surface by checking the sign of
    the dot product and flipping the normal if needed. Returns the
    reflected direction on total internal reflection.

    Args:
        direction: Ray direction (3,), normalized.
        normal: Surface normal (3,), normalized, pointing outward.
        n1: Refractive index of incident medium.
        n2: Refractive index of transmitted medium.

    Returns:
        refracted: Refracted direction (3,), or reflected if TIR.
        cos_theta_t: Cosine of refracted angle.
        tir: True if total internal reflection occurred.
    """
    cos_i = -jnp.dot(direction, normal)

    flip = cos_i < 0
    normal = jnp.where(flip, -normal, normal)
    cos_i = jnp.where(flip, -cos_i, cos_i)

    eta = n1 / n2
    sin2_t = eta**2 * (1.0 - cos_i**2)
    tir = sin2_t > 1.0

    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))
    refracted = eta * direction + (eta * cos_i - cos_t) * normal
    refracted = refracted / jnp.linalg.norm(refracted)

    reflected = direction - 2.0 * (-cos_i) * normal
    refracted = jnp.where(tir, reflected, refracted)

    return refracted, cos_t, tir


def fresnel_unpolarized(cos_theta_i, cos_theta_t, n1, n2):
    """Fresnel reflection and transmission coefficients for unpolarized light.

    Args:
        cos_theta_i: Cosine of incident angle.
        cos_theta_t: Cosine of transmitted angle.
        n1: Refractive index of incident medium.
        n2: Refractive index of transmitted medium.

    Returns:
        R: Reflectance (0 <= R <= 1).
        T: Transmittance (T = 1 - R).
    """
    rs_num = n1 * cos_theta_i - n2 * cos_theta_t
    rs_den = n1 * cos_theta_i + n2 * cos_theta_t
    rs_den = jnp.where(jnp.abs(rs_den) < 1e-10, 1e-10, rs_den)
    rs = (rs_num / rs_den)**2

    rp_num = n2 * cos_theta_i - n1 * cos_theta_t
    rp_den = n2 * cos_theta_i + n1 * cos_theta_t
    rp_den = jnp.where(jnp.abs(rp_den) < 1e-10, 1e-10, rp_den)
    rp = (rp_num / rp_den)**2

    R = 0.5 * (rs + rp)
    return R, 1.0 - R


def refract_slab(direction, normal, position, n_out, n_in, thickness):
    """Refract ray through a parallel-sided slab (window).

    Handles entry and exit refractions with Fresnel transmission at both
    surfaces.

    Args:
        direction: Ray direction (3,), normalized.
        normal: Front surface normal (3,), pointing outward.
        position: Entry point position (3,).
        n_out: Refractive index of ambient medium.
        n_in: Refractive index of slab material.
        thickness: Slab thickness.

    Returns:
        exit_direction: Direction after exiting slab (3,).
        exit_position: Position where ray exits slab (3,).
        transmittance: Combined Fresnel transmission for both surfaces.
        valid: True if ray transmitted (no TIR).
    """
    # Entry refraction
    dir_inside, cos_t_entry, tir_entry = refract(direction, normal, n_out, n_in)
    cos_i_entry = jnp.abs(jnp.dot(direction, normal))
    _, T_entry = fresnel_unpolarized(cos_i_entry, cos_t_entry, n_out, n_in)

    # Propagation through slab
    cos_to_normal = jnp.maximum(jnp.abs(jnp.dot(dir_inside, normal)), 1e-10)
    path_length = thickness / cos_to_normal
    exit_position = position + path_length * dir_inside

    # Exit refraction
    back_normal = -normal
    exit_direction, cos_t_exit, tir_exit = refract(dir_inside, back_normal, n_in, n_out)
    cos_i_exit = jnp.abs(jnp.dot(dir_inside, back_normal))
    _, T_exit = fresnel_unpolarized(cos_i_exit, cos_t_exit, n_in, n_out)

    valid = ~tir_entry & ~tir_exit
    transmittance = jnp.where(valid, T_entry * T_exit, 0.0)

    return exit_direction, exit_position, transmittance, valid


# Interaction modules

class ReflectInteraction(Interaction):
    """Reflection interaction for mirrors.

    Attributes:
        reflectivity: Per-element reflectivity coefficient (N,).
    """

    reflectivity: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.REFLECT

    def apply(self, directions, normals, points, element_idx):
        """Apply reflection at hit points.

        Returns:
            (new_directions, new_positions, coefficients)
        """
        reflected, _ = jax.vmap(reflect)(directions, normals)
        return reflected, points, self.reflectivity[element_idx]


class RefractInteraction(Interaction):
    """Single-surface refraction interaction for lenses.

    Attributes:
        n_inside: Refractive index of material per element (N,).
        n_outside: Ambient refractive index (scalar).
        transmittance: Bulk transmission coefficient per element (N,).
    """

    n_inside: Array       # (N,)
    n_outside: float
    transmittance: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.REFRACT

    def apply(self, directions, normals, points, element_idx):
        """Apply refraction at hit points (Snell's law + Fresnel).

        Returns:
            (new_directions, new_positions, coefficients)
        """
        n_in = self.n_inside[element_idx]
        n_out = self.n_outside

        def f(d, n, ni):
            refracted, cos_t, tir = refract(d, n, n_out, ni)
            cos_i = jnp.abs(jnp.dot(d, n))
            _, T = fresnel_unpolarized(cos_i, cos_t, n_out, ni)
            return refracted, jnp.where(tir, 0.0, T)

        refracted, fresnel_T = jax.vmap(f)(directions, normals, n_in)
        return refracted, points, self.transmittance[element_idx] * fresnel_T


class SlabInteraction(Interaction):
    """Parallel-sided slab (window) interaction.

    Attributes:
        n_inside: Refractive index of slab material per element (N,).
        n_outside: Ambient refractive index (scalar).
        thickness: Slab thickness per element (N,).
        transmittance: Bulk transmission coefficient per element (N,).
    """

    n_inside: Array       # (N,)
    n_outside: float
    thickness: Array      # (N,)
    transmittance: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.SLAB

    def apply(self, directions, normals, points, element_idx):
        """Apply slab refraction (entry + propagation + exit).

        Returns:
            (new_directions, new_positions, coefficients)
        """
        n_in = self.n_inside[element_idx]
        n_out = self.n_outside
        thick = self.thickness[element_idx]

        def f(d, n, pos, ni, th):
            ed, ep, tc, _ = refract_slab(d, n, pos, n_out, ni, th)
            return ed, ep, tc

        exit_dirs, exit_pos, tc = jax.vmap(f)(directions, normals, points, n_in, thick)
        return exit_dirs, exit_pos, self.transmittance[element_idx] * tc
