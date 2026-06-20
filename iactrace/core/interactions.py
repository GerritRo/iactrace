from __future__ import annotations

import enum
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .coatings import Coating, fresnel_unpolarized


def reflect(direction, normal):
    """Reflect a ray's direction off a surface.

    Args:
        direction: Ray direction (3,), pointing into the surface.
        normal: Surface normal (3,), pointing outward.

    Returns:
        reflected: Reflected direction (3,).
        cos_i: Cosine of the incidence angle (non-negative).
    """
    d_dot_n = jnp.sum(direction * normal, axis=-1, keepdims=True)
    reflected = direction - 2.0 * d_dot_n * normal
    return reflected, -d_dot_n


def refract(direction, normal, n1, n2):
    """Refract a ray's direction through an interface (Snell's law).

    Handles rays from either side of the surface by flipping the normal
    if needed. On total internal reflection, the *reflected* direction
    is returned in place of the refracted one and ``tir`` is ``True``.

    Args:
        direction: Ray direction (3,), normalized.
        normal: Surface normal (3,), normalized, pointing outward.
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmitted medium.

    Returns:
        refracted: Refracted direction (3,), or reflected if TIR.
        cos_i: Cosine of the incidence angle (non-negative, with the
            ambient-vs-internal side correctly resolved).
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

    return refracted, cos_i, tir


def refract_slab(direction, normal, position, n_out, n_in, thickness):
    """Refract a ray through a parallel-sided slab (window).

    Args:
        direction: Ray direction (3,), normalized.
        normal: Front-surface normal (3,), pointing outward.
        position: Entry point in world coordinates (3,).
        n_out: Refractive index of the ambient medium.
        n_in: Refractive index of the slab material.
        thickness: Slab thickness in the same units as ``position``.

    Returns:
        exit_direction: Ray direction after leaving the slab.
        exit_position: World-space point where the ray exits.
        cos_i: Cosine of the incidence angle on the slab from outside.
        valid: ``True`` iff no total internal reflection occurred at
            either face.
        path_length: Geometric distance the ray travels inside the
            slab, in the same units as ``thickness``. Multiplied by
            ``n_in`` by the caller to obtain the OPL contribution
            ``n_in * L``.
    """
    # Entry refraction; cos_i is the incidence cosine
    dir_inside, cos_i, tir_entry = refract(direction, normal, n_out, n_in)

    # Propagation through the bulk
    cos_to_normal = jnp.maximum(jnp.abs(jnp.dot(dir_inside, normal)), 1e-10)
    path_length = thickness / cos_to_normal
    exit_position = position + path_length * dir_inside

    # Exit refraction
    exit_direction, _, tir_exit = refract(dir_inside, -normal, n_in, n_out)

    return (
        exit_direction,
        exit_position,
        cos_i,
        ~tir_entry & ~tir_exit,
        path_length,
    )


# Interactions


class InteractionType(enum.Enum):
    """Type of optical interaction at a surface."""
    REFLECT = "reflect"
    REFRACT = "refract"
    SLAB = "slab"


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

        Returns a 5-tuple
        ``(new_directions, new_positions, coefficients, opl_internal, new_n)``.
        """


class ReflectInteraction(Interaction):
    """Reflection interaction for mirrors.

    Per-ray coefficient::

        reflectivity_scalar[idx] * reflectivity(cos_theta_i, idx)

    When ``reflectivity is None`` (the default) the angular factor is
    unity, i.e. an ideal angle-independent mirror with response
    ``reflectivity_scalar``. Provide a :class:`TabulatedCoating` (or
    any :class:`Coating`) to model a measured R(theta) curve. Reflection
    does not change the medium: ``new_n == current_n``.

    Attributes:
        reflectivity: Angle-dependent coating, or ``None`` for a flat
            angular response.
        reflectivity_scalar: Per-element bulk multiplier in ``[0, 1]``,
            shape ``(N,)``. Operations such as
            :func:`~iactrace.telescope.operations.set_reflectivity`
            write here, leaving the coating untouched.
    """

    reflectivity: Coating | None
    reflectivity_scalar: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.REFLECT

    def apply(self, directions, normals, points, element_idx, current_n):
        reflected, cos_array = jax.vmap(reflect)(directions, normals)
        cos_i = jnp.abs(cos_array.squeeze(-1))
        scalar = self.reflectivity_scalar[element_idx]
        if self.reflectivity is None:
            coeff = scalar
        else:
            coeff = scalar * self.reflectivity(cos_i, element_idx)
        opl_internal = jnp.zeros(directions.shape[0])
        return reflected, points, coeff, opl_internal, current_n


class RefractInteraction(Interaction):
    """Single-surface refraction interaction for lenses.

    Per-ray coefficient::

        transmittance_scalar[idx] * angular_response(cos_theta_i)

    When ``transmittance is None`` (the default) the angular response
    is :func:`~iactrace.core.coatings.fresnel_unpolarized` evaluated
    from ``current_n`` (the medium the ray is currently in) and
    ``n_inside`` (the far side of this surface). Snell's law is always
    applied to bend the ray.

    Semantically, this represents the ray crossing a single interface
    from one medium into another. A real glass body (e.g. a biconvex
    lens) is modelled as two consecutive :class:`RefractInteraction`
    stages, front then back surface, and the render loop's per-ray
    medium tracker carries the correct index through the glass
    interior between them, so OPL is exact.

    Attributes:
        n_inside: Refractive index on the far side of this surface,
            per element (N,). "Far side" means the medium the ray
            transmits *into*: for a front surface this is the glass
            index, for a back surface it is the ambient index.
        n_outside: Retained for back-compat / (de)serialisation
            (YAML schema). The render loop ignores it and uses the
            per-ray ``current_n`` as the incident-side index instead.
        transmittance: Angle-dependent coating, or ``None`` for
            bare-interface Fresnel transmittance.
        transmittance_scalar: Per-element bulk multiplier in ``[0, 1]``,
            shape ``(N,)``.
    """

    n_inside: Array
    n_outside: float
    transmittance: Coating | None
    transmittance_scalar: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.REFRACT

    def apply(self, directions, normals, points, element_idx, current_n):
        n_in = self.n_inside[element_idx]

        refracted, cos_i, tir = jax.vmap(refract)(
            directions, normals, current_n, n_in,
        )

        if self.transmittance is None:
            _, t = fresnel_unpolarized(cos_i, current_n, n_in)
        else:
            t = self.transmittance(cos_i, element_idx)
        t = jnp.where(tir, 0.0, t)

        opl_internal = jnp.zeros(directions.shape[0])
        coeff = self.transmittance_scalar[element_idx] * t
        return refracted, points, coeff, opl_internal, n_in


class SlabInteraction(Interaction):
    """Parallel-sided slab (window) interaction.

    Per-ray coefficient::

        transmittance_scalar[idx] * angular_response

    When ``transmittance is None`` (the default) the angular response
    is the standard Fresnel product at the two faces: by parallel-slab
    symmetry and Stokes reciprocity, both faces share the same
    single-face Fresnel coefficient so the result simplifies to
    ``T_face^2``. Provide a :class:`Coating` to override with a vendor-
    supplied T(theta) curve for the *complete* slab; the coating fully
    replaces the Fresnel product. The TIR mask from the underlying
    geometry gates out invalid rays either way.

    The ray enters from its current medium (``current_n``), refracts
    into the slab material, traverses it, and refracts back out into
    the *same* medium; slabs assume the ambient is symmetric across
    them, which is the usual case for a window. ``opl_internal`` is
    the per-ray ``n_in * L`` inside the slab.

    Attributes:
        n_inside: Per-element slab refractive index, shape ``(N,)``.
        n_outside: Retained for back-compat / (de)serialisation. The
            render loop ignores it and uses the per-ray ``current_n``
            for both entry and exit refraction.
        thickness: Per-element slab thickness, shape ``(N,)``.
        transmittance: Angle-dependent coating, or ``None`` for the
            bare-window Fresnel product.
        transmittance_scalar: Per-element bulk multiplier in ``[0, 1]``,
            shape ``(N,)``.
    """

    n_inside: Array
    n_outside: float
    thickness: Array
    transmittance: Coating | None
    transmittance_scalar: Array  # (N,)

    @property
    def interaction_type(self) -> InteractionType:
        return InteractionType.SLAB

    def apply(self, directions, normals, points, element_idx, current_n):
        n_in = self.n_inside[element_idx]
        thick = self.thickness[element_idx]

        exit_dir, exit_pos, cos_i, valid, path_length = jax.vmap(
            lambda d, n, pos, n_amb, ni, th: refract_slab(
                d, n, pos, n_amb, ni, th,
            )
        )(directions, normals, points, current_n, n_in, thick)

        if self.transmittance is None:
            _, T_face = fresnel_unpolarized(cos_i, current_n, n_in)
            t = T_face * T_face
        else:
            t = self.transmittance(cos_i, element_idx)
        t = jnp.where(valid, t, 0.0)

        opl_internal = jnp.where(valid, n_in * path_length, 0.0)
        coeff = self.transmittance_scalar[element_idx] * t
        return exit_dir, exit_pos, coeff, opl_internal, current_n
