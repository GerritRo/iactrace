from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from ..core.apertures import Aperture, DiskAperture
from ..core.coatings import Coating
from ..core.interactions import RefractInteraction, SlabInteraction
from ..core.optics import OpticalElementGroup
from ..core.surfaces import AsphericSurfaceGroup
from ._common import as_aspheric_row as _as_aspheric_row
from ._common import as_vec3 as _as_vec3

__all__ = [
    "refractive_group",
    "slab_group",
    "aspheric_lens",
    "plano_slab",
]


# Low-level canonical builders


def refractive_group(
    *,
    positions: Array,
    rotations: Array,
    curvatures: Array,
    conics: Array,
    aspherics: Array,
    offsets: Array,
    aperture: Aperture,
    n_inside: Array,
    transmittance: Array | float = 1.0,
    sample_key: Array,
    coating: Coating | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
) -> OpticalElementGroup:
    """Canonical builder for refractive :class:`OpticalElementGroup` instances.

    Assembles an :class:`AsphericSurfaceGroup` + :class:`RefractInteraction`
    + group wiring from pre-shaped arrays. Use this for curved single-surface
    refracting lenses with any aperture type. Both the user-facing sugar
    helpers and the YAML adapter delegate to this function.

    Args:
        positions: Per-element vertex positions, shape ``(N, 3)``.
        rotations: Per-element Euler angles in degrees, shape ``(N, 3)``.
        curvatures: Per-element curvatures ``1/R``, shape ``(N,)``.
        conics: Per-element conic constants, shape ``(N,)``.
        aspherics: Per-element even aspheric coefficients ``[A4, A6, ...]``,
            shape ``(N, K)``; column ``i`` multiplies ``r^(2i + 4)``.
        offsets: Per-element surface decentering, shape ``(N, 2)``.
        aperture: Pre-built aperture sized to ``N``.
        n_inside: Per-element refractive index, shape ``(N,)``. The ambient
            index on the incident side is not stored: the render loop reads
            it dynamically from each ray's current medium (see
            :class:`~iactrace.core.interactions.RefractInteraction`).
        transmittance: Per-element bulk transmittance in ``[0, 1]``, shape
            ``(N,)``.
        sample_key: JAX PRNG key for aperture sampling.
        optical_stage: Stage index within the Telescope.
        n_samples: Monte Carlo samples per element per render.
    """
    positions = jnp.asarray(positions)
    rotations = jnp.asarray(rotations)
    curvatures = jnp.asarray(curvatures)
    conics = jnp.asarray(conics)
    aspherics = jnp.asarray(aspherics)
    offsets = jnp.asarray(offsets)
    n_inside = jnp.asarray(n_inside)
    n = int(positions.shape[0])

    trans_scalar = jnp.asarray(transmittance)
    if trans_scalar.ndim == 0:
        trans_scalar = jnp.full((n,), trans_scalar)

    surface = AsphericSurfaceGroup(
        curvatures=curvatures,
        conics=conics,
        aspherics=aspherics,
        offsets=offsets,
    )
    interaction = RefractInteraction(
        n_inside=n_inside,
        transmittance=coating,
        transmittance_scalar=trans_scalar,
    )

    return OpticalElementGroup(
        positions=positions,
        rotations=rotations,
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=sample_key,
        optical_stage=int(optical_stage),
        n_samples=int(n_samples),
    )


def slab_group(
    *,
    positions: Array,
    rotations: Array,
    aperture: Aperture,
    n_inside: Array,
    thickness: Array,
    transmittance: Array | float = 1.0,
    sample_key: Array,
    coating: Coating | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
) -> OpticalElementGroup:
    """Canonical builder for parallel-sided slab (window) groups.

    The surface is always zero curvature, conic, and aspheric, a slab is
    by definition flat, so no surface parameters are exposed. Both the
    user-facing :func:`plano_slab` helper and the YAML adapter delegate
    to this function.

    Args:
        positions: Per-element front-surface positions, shape ``(N, 3)``.
        rotations: Per-element Euler angles in degrees, shape ``(N, 3)``.
        aperture: Pre-built aperture sized to ``N``.
        n_inside: Per-element slab refractive index, shape ``(N,)``. The
            ambient index is not stored: the render loop reads it
            dynamically from each ray's current medium (see
            :class:`~iactrace.core.interactions.SlabInteraction`).
        thickness: Per-element slab thickness in metres, shape ``(N,)``.
        transmittance: Per-element bulk transmittance in ``[0, 1]``, shape
            ``(N,)``.
        sample_key: JAX PRNG key for aperture sampling.
        optical_stage: Stage index within the Telescope.
        n_samples: Monte Carlo samples per element per render.
    """
    positions = jnp.asarray(positions)
    rotations = jnp.asarray(rotations)
    n_inside = jnp.asarray(n_inside)
    thickness = jnp.asarray(thickness)
    n = int(positions.shape[0])

    trans_scalar = jnp.asarray(transmittance)
    if trans_scalar.ndim == 0:
        trans_scalar = jnp.full((n,), trans_scalar)

    surface = AsphericSurfaceGroup(
        curvatures=jnp.zeros(n),
        conics=jnp.zeros(n),
        aspherics=jnp.zeros((n, 0)),
        offsets=jnp.zeros((n, 2)),
    )
    interaction = SlabInteraction(
        n_inside=n_inside,
        thickness=thickness,
        transmittance=coating,
        transmittance_scalar=trans_scalar,
    )

    return OpticalElementGroup(
        positions=positions,
        rotations=rotations,
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=sample_key,
        optical_stage=int(optical_stage),
        n_samples=int(n_samples),
    )


# High-level sugar: single-element factories


def _single_disk_aperture(radius: float) -> DiskAperture:
    """A solid (no inner hole) ``DiskAperture`` for a single-element group."""
    return DiskAperture(radii=jnp.asarray([float(radius)]), inner_radii=jnp.zeros(1))


def _single_disk_refractive(
    *,
    position,
    rotation,
    curvature,
    conic,
    aspheric_coeffs,
    radius,
    n_inside,
    transmittance,
    coating,
    optical_stage,
    n_samples,
    key,
) -> OpticalElementGroup:
    """Common backing for :func:`thin` and :func:`aspheric_lens`."""
    pos = _as_vec3(position, "position")
    rot = _as_vec3(rotation, "rotation")
    aspheric_row = _as_aspheric_row(aspheric_coeffs)
    n = 1

    aperture = _single_disk_aperture(radius)

    return refractive_group(
        positions=pos.reshape(1, 3),
        rotations=rot.reshape(1, 3),
        curvatures=jnp.asarray([float(curvature)]),
        conics=jnp.asarray([float(conic)]),
        aspherics=aspheric_row.reshape(1, aspheric_row.shape[0]),
        offsets=jnp.zeros((n, 2)),
        aperture=aperture,
        n_inside=jnp.asarray([float(n_inside)]),
        transmittance=jnp.asarray([float(transmittance)]),
        coating=coating,
        sample_key=key,
        optical_stage=optical_stage,
        n_samples=n_samples,
    )


def aspheric_lens(
    *,
    position: Sequence[float],
    curvature: float,
    radius: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    conic: float = 0.0,
    aspheric_coeffs: Sequence[float] | None = None,
    n_inside: float = 1.5,
    transmittance: float = 1.0,
    coating: Coating | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a general aspheric-disk refracting lens as a single-element group.

    Fully explicit version of :func:`thin`: you supply the raw curvature,
    conic constant and optional aspheric coefficients. Use this for
    precisely-specified lens prescriptions.

    Args:
        position: Vertex position in world coordinates, shape (3,).
        curvature: Surface curvature ``1/R`` in m^-1.
        radius: Outer disk radius in metres.
        rotation: Euler angles in degrees. Defaults to no rotation.
        conic: Schwarzschild conic constant.
        aspheric_coeffs: Even aspheric coefficients ``[A4, A6, ...]``,
            i.e. ``aspheric_coeffs[i]`` multiplies ``r^(2i + 4)``.
        n_inside, transmittance, optical_stage, n_samples, key:
            see :func:`thin`.
    """
    return _single_disk_refractive(
        position=position,
        rotation=rotation,
        curvature=curvature,
        conic=conic,
        aspheric_coeffs=aspheric_coeffs,
        radius=radius,
        n_inside=n_inside,
        transmittance=transmittance,
        coating=coating,
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )


def plano_slab(
    *,
    position: Sequence[float],
    radius: float,
    thickness: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    n_inside: float = 1.5,
    transmittance: float = 1.0,
    coating: Coating | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a flat parallel-sided window as a single-element group.

    The surface is zero-curvature (flat) and rays pass through a
    :class:`SlabInteraction` that handles entry refraction, propagation
    through the slab, and exit refraction. Useful for entrance windows
    and filters.

    Args:
        position: Vertex position of the front surface, shape (3,).
        radius: Outer disk radius in metres.
        thickness: Slab thickness in metres (along the optical axis).
        rotation: Euler angles in degrees. Defaults to no rotation.
        n_inside: Refractive index of the slab material. Defaults to 1.5.
        transmittance: Bulk transmittance in ``[0, 1]``. Defaults to 1.0.
        optical_stage: Stage index within the Telescope.
        n_samples: Monte Carlo samples per element per render.
        key: JAX PRNG key for aperture sampling.
    """
    pos = _as_vec3(position, "position")
    rot = _as_vec3(rotation, "rotation")

    aperture = _single_disk_aperture(radius)

    return slab_group(
        positions=pos.reshape(1, 3),
        rotations=rot.reshape(1, 3),
        aperture=aperture,
        n_inside=jnp.asarray([float(n_inside)]),
        thickness=jnp.asarray([float(thickness)]),
        transmittance=jnp.asarray([float(transmittance)]),
        coating=coating,
        sample_key=key,
        optical_stage=optical_stage,
        n_samples=n_samples,
    )
