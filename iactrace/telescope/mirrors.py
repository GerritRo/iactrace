from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..core.apertures import Aperture, DiskAperture
from ..core.bsdf import BSDF, GaussianBSDF
from ..core.coatings import Coating
from ..core.interactions import ReflectInteraction
from ..core.optics import OpticalElementGroup
from ..core.surfaces import AsphericSurfaceGroup
from ._common import as_aspheric_row as _as_aspheric_row
from ._common import as_vec3 as _as_vec3

__all__ = [
    "mirror_group",
    "spherical",
    "parabolic",
    "aspheric",
    "disk_array",
]


# Low-level canonical builder


def mirror_group(
    *,
    positions: Array,
    rotations: Array,
    curvatures: Array,
    conics: Array,
    aspherics: Array,
    offsets: Array,
    aperture: Aperture,
    reflectivity: Array | float = 1.0,
    sample_key: Array,
    coating: Coating | None = None,
    bsdf: BSDF | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
) -> OpticalElementGroup:
    """Canonical reflective :class:`OpticalElementGroup` builder.

    Takes pre-shaped per-element arrays plus a pre-built aperture and an
    optional :class:`BSDF` instance, and assembles the surface + interaction
    + group wiring.

    Args:
        positions: Per-element vertex positions, shape ``(N, 3)``.
        rotations: Per-element Euler angles in degrees, shape ``(N, 3)``.
        curvatures: Per-element curvatures ``1/R``, shape ``(N,)``.
        conics: Per-element Schwarzschild conic constants, shape ``(N,)``.
        aspherics: Per-element even aspheric coefficients ``[A4, A6, ...]``,
            shape ``(N, K)``; column ``i`` multiplies ``r^(2i + 4)``.
        offsets: Per-element surface decentering, shape ``(N, 2)``. Use
            ``jnp.zeros((N, 2))`` for a centred disk.
        aperture: Pre-built aperture.
        reflectivity: Per-element reflectivity in ``[0, 1]``, shape ``(N,)``.
        sample_key: JAX PRNG key used for aperture sampling and BSDF.
        bsdf: Optional :class:`BSDF` instance. ``None`` leaves the element
            perfectly specular (the :class:`OpticalElementGroup` constructor
            fills in a zero-scale :class:`GaussianBSDF`).
        optical_stage: Stage index within the Telescope; each group in a
            telescope must have a unique stage.
        n_samples: Monte Carlo samples per element per render.

    Returns:
        A ready-to-use :class:`OpticalElementGroup`.
    """
    positions = jnp.asarray(positions)
    rotations = jnp.asarray(rotations)
    curvatures = jnp.asarray(curvatures)
    conics = jnp.asarray(conics)
    aspherics = jnp.asarray(aspherics)
    offsets = jnp.asarray(offsets)
    n = int(positions.shape[0])

    refl_scalar = jnp.asarray(reflectivity)
    if refl_scalar.ndim == 0:
        refl_scalar = jnp.full((n,), refl_scalar)

    surface = AsphericSurfaceGroup(
        curvatures=curvatures,
        conics=conics,
        aspherics=aspherics,
        offsets=offsets,
    )
    interaction = ReflectInteraction(
        reflectivity=coating,
        reflectivity_scalar=refl_scalar,
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
        bsdf=bsdf,
    )


# High-level sugar: batched disk-aperture mirror group


def disk_array(
    *,
    positions: ArrayLike,
    rotations: ArrayLike,
    curvatures: ArrayLike,
    radii: ArrayLike,
    conics: ArrayLike | None = None,
    aspheric_coeffs: ArrayLike | None = None,
    inner_radii: ArrayLike | None = None,
    reflectivities: ArrayLike | None = None,
    coating: Coating | None = None,
    bsdf_scales: ArrayLike | None = None,
    offsets: ArrayLike | None = None,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a batched ``N``-element disk-aperture mirror group.

    Use this for segmented primary mirrors. Per-element arrays must all
    match length ``N``. Scalar defaults fill in where an argument is
    omitted. For anything beyond disk apertures / Gaussian BSDF, drop
    down to :func:`mirror_group`.

    Args:
        positions: Per-element vertex positions, shape ``(N, 3)``.
        rotations: Per-element Euler angles in degrees, shape ``(N, 3)``.
        curvatures: Per-element curvatures ``1/R``, shape ``(N,)``.
        radii: Outer disk radii, shape ``(N,)``.
        conics: Per-element conic constants, shape ``(N,)``. Defaults to
            zeros (spherical).
        aspheric_coeffs: Per-element aspheric coefficients, shape ``(N, K)``.
            ``None`` disables aspherics.
        inner_radii: Per-element central hole radii, shape ``(N,)``.
            Defaults to zeros.
        reflectivities: Per-element reflectivities, shape ``(N,)``. Defaults
            to ones.
        bsdf_scales: Per-element Gaussian BSDF roughness in arcseconds,
            shape ``(N,)``. Zero (the default) disables the BSDF.
        offsets: Per-element surface decentering, shape ``(N, 2)``. Defaults
            to zeros.
        optical_stage: Stage index shared by all elements in this group.
        n_samples: Monte Carlo samples per element per render.
        key: JAX PRNG key for aperture sampling and BSDF.
    """
    positions_arr = jnp.asarray(positions)
    if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions_arr.shape}")
    n = positions_arr.shape[0]

    rotations_arr = jnp.asarray(rotations)
    if rotations_arr.shape != (n, 3):
        raise ValueError(f"rotations must have shape ({n}, 3), got {rotations_arr.shape}")

    curvatures_arr = jnp.asarray(curvatures)
    if curvatures_arr.shape != (n,):
        raise ValueError(f"curvatures must have shape ({n},), got {curvatures_arr.shape}")

    radii_arr = jnp.asarray(radii)
    if radii_arr.shape != (n,):
        raise ValueError(f"radii must have shape ({n},), got {radii_arr.shape}")

    conics_arr = jnp.zeros(n) if conics is None else jnp.asarray(conics)
    if conics_arr.shape != (n,):
        raise ValueError(f"conics must have shape ({n},), got {conics_arr.shape}")

    if aspheric_coeffs is None:
        aspherics_arr = jnp.zeros((n, 0))
    else:
        aspherics_arr = jnp.asarray(aspheric_coeffs)
        if aspherics_arr.ndim != 2 or aspherics_arr.shape[0] != n:
            raise ValueError(f"aspheric_coeffs must have shape ({n}, K), got {aspherics_arr.shape}")

    inner_arr = jnp.zeros(n) if inner_radii is None else jnp.asarray(inner_radii)
    refl_arr = jnp.ones(n) if reflectivities is None else jnp.asarray(reflectivities)

    if offsets is None:
        offsets_arr = jnp.zeros((n, 2))
    else:
        offsets_arr = jnp.asarray(offsets)
        if offsets_arr.shape != (n, 2):
            raise ValueError(f"offsets must have shape ({n}, 2), got {offsets_arr.shape}")

    bsdf_arr = jnp.zeros(n) if bsdf_scales is None else jnp.asarray(bsdf_scales)
    bsdf = None if bool(jnp.all(bsdf_arr == 0)) else GaussianBSDF(scale=bsdf_arr)

    aperture = DiskAperture(radii=radii_arr, inner_radii=inner_arr)

    return mirror_group(
        positions=positions_arr,
        rotations=rotations_arr,
        curvatures=curvatures_arr,
        conics=conics_arr,
        aspherics=aspherics_arr,
        offsets=offsets_arr,
        aperture=aperture,
        reflectivity=refl_arr,
        coating=coating,
        bsdf=bsdf,
        sample_key=key,
        optical_stage=optical_stage,
        n_samples=n_samples,
    )


# High-level sugar: single-element factories


def _single_disk_mirror(
    *,
    position,
    rotation,
    curvature,
    conic,
    aspheric_coeffs,
    radius,
    inner_radius,
    reflectivity,
    coating,
    bsdf_scale,
    optical_stage,
    n_samples,
    key,
) -> OpticalElementGroup:
    """Common backing for :func:`spherical`, :func:`parabolic`, :func:`aspheric`."""
    pos = _as_vec3(position, "position")
    rot = _as_vec3(rotation, "rotation")
    aspheric_row = _as_aspheric_row(aspheric_coeffs)

    return disk_array(
        positions=pos.reshape(1, 3),
        rotations=rot.reshape(1, 3),
        curvatures=jnp.asarray([float(curvature)]),
        conics=jnp.asarray([float(conic)]),
        aspheric_coeffs=aspheric_row.reshape(1, aspheric_row.shape[0]),
        radii=jnp.asarray([float(radius)]),
        inner_radii=jnp.asarray([float(inner_radius)]),
        reflectivities=jnp.asarray([float(reflectivity)]),
        coating=coating,
        bsdf_scales=jnp.asarray([float(bsdf_scale)]),
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )


def _focal_length_disk_mirror(
    *,
    conic: float,
    position: Sequence[float],
    focal_length: float,
    radius: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    inner_radius: float = 0.0,
    reflectivity: float = 1.0,
    coating: Coating | None = None,
    bsdf_scale: float = 0.0,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Common backing for :func:`spherical` and :func:`parabolic`.

    Both derive ``curvature = 1 / (2 * focal_length)`` and differ only in
    ``conic`` (``0`` vs ``-1``).
    """
    return _single_disk_mirror(
        position=position,
        rotation=rotation,
        curvature=1.0 / (2.0 * float(focal_length)),
        conic=conic,
        aspheric_coeffs=None,
        radius=radius,
        inner_radius=inner_radius,
        reflectivity=reflectivity,
        coating=coating,
        bsdf_scale=bsdf_scale,
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )


def spherical(
    *,
    position: Sequence[float],
    focal_length: float,
    radius: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    inner_radius: float = 0.0,
    reflectivity: float = 1.0,
    coating: Coating | None = None,
    bsdf_scale: float = 0.0,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a spherical mirror as a single-element group.

    Uses ``c = 1 / (2 * focal_length)`` with ``conic = 0``. Set
    ``inner_radius > 0`` for an annular mirror.

    Args:
        position: Mirror vertex in world coordinates, shape (3,).
        focal_length: Paraxial focal length in metres (positive = concave).
        radius: Outer disk radius in metres.
        rotation: Euler angles in degrees. Defaults to no rotation.
        inner_radius: Inner hole radius in metres. Zero for a solid disk.
        reflectivity: Per-element reflectivity in ``[0, 1]``.
        bsdf_scale: Gaussian roughness sigma in arcseconds (0 disables).
        optical_stage: Stage index within the Telescope.
        n_samples: Monte Carlo samples per render call.
        key: JAX PRNG key.
    """
    return _focal_length_disk_mirror(
        conic=0.0,
        position=position,
        focal_length=focal_length,
        radius=radius,
        rotation=rotation,
        inner_radius=inner_radius,
        reflectivity=reflectivity,
        coating=coating,
        bsdf_scale=bsdf_scale,
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )


def parabolic(
    *,
    position: Sequence[float],
    focal_length: float,
    radius: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    inner_radius: float = 0.0,
    reflectivity: float = 1.0,
    coating: Coating | None = None,
    bsdf_scale: float = 0.0,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a parabolic mirror as a single-element group.

    Uses ``c = 1 / (2 * focal_length)`` and ``conic = -1``, matching the
    reference ``configs/BASIC/Cassegrain_telescope.yaml`` primary
    (``focal_length=0.4`` -> ``curvature=1.25``).

    Args: see :func:`spherical`.
    """
    return _focal_length_disk_mirror(
        conic=-1.0,
        position=position,
        focal_length=focal_length,
        radius=radius,
        rotation=rotation,
        inner_radius=inner_radius,
        reflectivity=reflectivity,
        coating=coating,
        bsdf_scale=bsdf_scale,
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )


def aspheric(
    *,
    position: Sequence[float],
    curvature: float,
    radius: float,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    conic: float = 0.0,
    aspheric_coeffs: Sequence[float] | None = None,
    inner_radius: float = 0.0,
    reflectivity: float = 1.0,
    coating: Coating | None = None,
    bsdf_scale: float = 0.0,
    optical_stage: int = 0,
    n_samples: int = 100,
    key: Array,
) -> OpticalElementGroup:
    """Build a general aspheric mirror as a single-element group.

    Fully explicit version of :func:`spherical` / :func:`parabolic`: you
    supply the curvature, conic constant, and optional even aspheric
    coefficients. Use this for hyperbolic / elliptic / higher-order
    aspheric mirrors.

    Args:
        position: Mirror vertex in world coordinates, shape (3,).
        curvature: Paraxial curvature ``1/R`` in m^-1.
        radius: Outer disk radius in metres.
        rotation: Euler angles in degrees. Defaults to no rotation.
        conic: Schwarzschild conic constant. ``0`` spherical, ``-1``
            parabolic, ``-1 < k < 0`` prolate ellipsoid, ``k < -1``
            hyperboloid.
        aspheric_coeffs: Even aspheric coefficients ``[A4, A6, ...]``,
            i.e. ``aspheric_coeffs[i]`` multiplies ``r^(2i + 4)``. The
            polynomial starts at ``r^4``.
        inner_radius, reflectivity, bsdf_scale, optical_stage, n_samples, key:
            see :func:`spherical`.
    """
    return _single_disk_mirror(
        position=position,
        rotation=rotation,
        curvature=curvature,
        conic=conic,
        aspheric_coeffs=aspheric_coeffs,
        radius=radius,
        inner_radius=inner_radius,
        reflectivity=reflectivity,
        coating=coating,
        bsdf_scale=bsdf_scale,
        optical_stage=optical_stage,
        n_samples=n_samples,
        key=key,
    )
