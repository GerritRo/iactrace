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
    def apply(self, directions, normals, points, element_idx, current_n):
        """Apply the interaction at hit points.

        Args:
            directions, normals, points, element_idx: per-ray geometry
                at the surface hit.
            current_n: per-ray refractive index of the medium the ray
                is currently propagating in. Used as the incident-side
                index for refraction physics, so OPL is exact even
                through stacked refractive surfaces.

        Returns a 5-tuple ``(new_directions, new_positions, coefficients,
        opl_internal, new_n)``:

        * ``opl_internal`` is the per-ray optical path length accumulated
          *inside* the interaction itself (non-zero only for slab /
          window elements, which carry an internal ``n·L`` segment).
        * ``new_n`` is the per-ray refractive index of the medium the
          ray exits into. The render loop uses it to weight the next
          inter-stage segment.
        """


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
        opl_inside: Optical path length traversed inside the slab,
            ``n_in * thickness / cos_to_normal``. Zero when the ray is
            lost to total internal reflection.
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
    opl_inside = jnp.where(valid, n_in * path_length, 0.0)

    return exit_direction, exit_position, transmittance, valid, opl_inside


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

    def apply(self, directions, normals, points, element_idx, current_n):
        """Apply reflection at hit points.

        Reflection does not change the medium — ``new_n == current_n``.
        """
        reflected, _ = jax.vmap(reflect)(directions, normals)
        opl_internal = jnp.zeros(directions.shape[0])
        return reflected, points, self.reflectivity[element_idx], opl_internal, current_n


class RefractInteraction(Interaction):
    """Single-surface refraction interaction for lenses.

    Semantically, this represents the ray crossing a single interface
    from one medium into another. The incident-side index is the
    per-ray tracked medium (``current_n``); ``n_inside`` is the index
    on the far side of this surface. A real glass body (e.g. a
    biconvex lens) is modelled as two consecutive
    :class:`RefractInteraction` stages — front then back surface — and
    the render loop's per-ray medium tracker carries the correct index
    through the glass interior between them, so OPL is exact.

    Attributes:
        n_inside: Refractive index on the far side of this surface,
            per element (N,). "Far side" means the medium the ray
            transmits *into*: for a front surface this is the glass
            index, for a back surface it is the ambient index.
        n_outside: Retained for back-compat and (de)serialisation
            (YAML schema). The render loop ignores it and uses the
            per-ray tracked medium index instead, so OPL is correct
            regardless of what was written here.
        transmittance: Bulk transmission coefficient per element (N,).
    """

    n_inside: Array       # (N,)
    n_outside: float
    transmittance: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.REFRACT

    def apply(self, directions, normals, points, element_idx, current_n):
        """Apply refraction at hit points (Snell's law + Fresnel).

        Uses ``current_n`` (the per-ray tracked medium index) as the
        incident-side index and ``self.n_inside[element_idx]`` as the
        transmitted-side index. Returns ``new_n = n_inside[element_idx]``
        so the ray's medium follows it through the next segment.
        """
        n_in = self.n_inside[element_idx]

        def f(d, n, n1, n2):
            refracted, cos_t, tir = refract(d, n, n1, n2)
            cos_i = jnp.abs(jnp.dot(d, n))
            _, T = fresnel_unpolarized(cos_i, cos_t, n1, n2)
            return refracted, jnp.where(tir, 0.0, T)

        refracted, fresnel_T = jax.vmap(f)(directions, normals, current_n, n_in)
        opl_internal = jnp.zeros(directions.shape[0])
        return refracted, points, self.transmittance[element_idx] * fresnel_T, opl_internal, n_in


class SlabInteraction(Interaction):
    """Parallel-sided slab (window) interaction.

    The ray enters from its current medium (``current_n``), refracts
    into the slab material, traverses it, and refracts back out into
    the *same* medium — slabs assume the ambient is symmetric across
    them, which is the usual case for a window.

    Attributes:
        n_inside: Refractive index of slab material per element (N,).
        n_outside: Retained for back-compat / (de)serialisation. The
            render loop ignores it and uses the per-ray tracked medium
            index for both entry and exit refraction.
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

    def apply(self, directions, normals, points, element_idx, current_n):
        """Apply slab refraction (entry + propagation + exit).

        ``opl_internal[i] = n_in * thickness / cos(theta_inside)`` is
        the optical path length traversed inside the slab glass; zero
        for rays lost to total internal reflection. The slab exits
        back into the incoming medium, so ``new_n = current_n``.
        """
        n_in = self.n_inside[element_idx]
        thick = self.thickness[element_idx]

        def f(d, n, pos, n_amb, ni, th):
            ed, ep, tc, _, opl = refract_slab(d, n, pos, n_amb, ni, th)
            return ed, ep, tc, opl

        exit_dirs, exit_pos, tc, opl_internal = jax.vmap(f)(
            directions, normals, points, current_n, n_in, thick,
        )
        return (
            exit_dirs, exit_pos,
            self.transmittance[element_idx] * tc,
            opl_internal, current_n,
        )