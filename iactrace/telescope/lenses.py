from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.optics import InteractionType, OpticalGroupBase
from ..core.transforms import euler_to_matrix


def _transform_disk_geometry_to_world(aperture_samples, offsets, curvatures, conics,
                                       aspherics, radii, perturbation_angles,
                                       positions, rotations, perturbation_scale):
    """Transform disk aperture geometry from local to world coordinates.

    Shared helper for all disk-aperture refractive elements (lenses and slabs).

    Args:
        aperture_samples: 2D sample positions on aperture (N, M, 2)
        offsets: Per-element offsets (N, 2)
        curvatures: Per-element curvatures (N,)
        conics: Per-element conic constants (N,)
        aspherics: Per-element aspheric coefficients (N, K)
        radii: Per-element aperture radii (N,)
        perturbation_angles: Random angles for roughness (N, M, 2)
        positions: Element positions (N, 3)
        rotations: Element rotations (N, 3)
        perturbation_scale: Roughness scale per element (N,)

    Returns:
        Tuple of (points_world, normals_world, weights):
            - points_world: (N, M, 3) world-space sample points
            - normals_world: (N, M, 3) world-space surface normals
            - weights: (N, M, 1) geometry integration weights
    """
    from ..core.optics import apply_perturbation
    from ..core.surfaces import compute_sag_and_normal

    def compute_and_transform_single(xy, offset, curvature, conic, aspheric, radius,
                                      angles, position, rotation, scale):
        x, y = xy[..., 0], xy[..., 1]
        points, normals = jax.vmap(
            lambda xi, yi: compute_sag_and_normal(xi, yi, offset, curvature, conic, aspheric)
        )(x, y)

        # Compute weights: cos(angle to z-axis) / area * n_samples
        cos_z = jnp.sum(normals * jnp.array([0.0, 0.0, 1.0]), axis=-1, keepdims=True)
        n_samples = xy.shape[0]
        area = jnp.pi * radius**2
        weights = cos_z / area * n_samples

        # Transform to world coordinates
        rot = euler_to_matrix(rotation)
        points_world = jnp.einsum('ij,nj->ni', rot, points) + position
        normals_world = jnp.einsum('ij,nj->ni', rot, normals)

        # Apply perturbation
        perturbed = apply_perturbation(normals_world, angles, scale)

        return points_world, perturbed, weights

    return jax.vmap(compute_and_transform_single)(
        aperture_samples, offsets, curvatures, conics, aspherics, radii,
        perturbation_angles, positions, rotations, perturbation_scale
    )


class LensGroup(OpticalGroupBase):
    """Base class for grouped refractive elements with shared aperture type.

    Defines the interface for lens groups, similar to MirrorGroup for mirrors.
    Concrete implementations (AsphericDiskLensGroup, PlanoSlabGroup) define
    the specific surface type and aperture geometry.

    Required attributes (defined by subclasses):
        positions: (N, 3) center positions
        rotations: (N, 3) Euler angles in degrees
        n_inside: (N,) refractive index of material
        n_outside: float, ambient refractive index (default 1.0)
        transmittance: (N,) bulk transmission coefficient
        optical_stage: int, 0=primary, 1=secondary, etc.
    """

    config_type: ClassVar[str] = ""  # Set by subclasses

    # Common attributes for all lens groups (no defaults - defined by subclasses)
    positions: jax.Array      # (N, 3)
    rotations: jax.Array      # (N, 3)
    n_inside: jax.Array       # (N,)
    transmittance: jax.Array  # (N,)

    # Sampling state
    aperture_samples: jax.Array       # (N, M, 2)
    perturbation_angles: jax.Array    # (N, M, 2)
    perturbation_scale: jax.Array     # (N,)
    perturbation_key: jax.Array

    # Fields with defaults must come after fields without defaults
    n_outside: float = 1.0
    optical_stage: int = eqx.field(static=True, default=0)

    @property
    @abstractmethod
    def interaction(self) -> InteractionType:
        """Return the type of optical interaction (REFRACT or SLAB)."""
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

    @abstractmethod
    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates.

        Returns:
            Tuple with at minimum (points_world, normals_world, weights, n_inside, transmittance).
            SLAB types also return thickness.
        """
        ...

    @abstractmethod
    def to_config(self, index: int) -> dict[str, Any]:
        """Convert a single lens element at index to a config dict."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, configs: list[dict[str, Any]], **kwargs: Any) -> "LensGroup":
        """Create a LensGroup from a list of config dicts."""
        ...

    def __len__(self):
        """Return number of elements in group."""
        return self.positions.shape[0]


class AsphericDiskLensGroup(LensGroup):
    """Group of refractive elements with aspheric surfaces and circular apertures.

    This models a single refractive surface (e.g., one side of a lens). For a
    complete lens with two surfaces, use two AsphericDiskLensGroup instances.
    For parallel-sided windows, use PlanoSlabGroup instead.
    """

    config_type: ClassVar[str] = "aspheric_disk"

    positions: jax.Array      # (N, 3)
    rotations: jax.Array      # (N, 3) euler angles in degrees
    curvatures: jax.Array     # (N,)
    conics: jax.Array         # (N,)
    aspherics: jax.Array      # (N, K_max)
    offsets: jax.Array        # (N, 2)
    radii: jax.Array          # (N,)
    n_inside: jax.Array       # (N,)
    transmittance: jax.Array  # (N,)
    aperture_samples: jax.Array   # (N, M, 2)
    perturbation_angles: jax.Array  # (N, M, 2)
    perturbation_scale: jax.Array  # (N,)
    perturbation_key: jax.Array

    n_outside: float
    optical_stage: int = eqx.field(static=True)
    is_pure_conic: bool = eqx.field(static=True)  # True if all aspherics are zero

    def __init__(self, positions, rotations, curvatures, conics, aspherics, radii,
                 n_inside, optical_stage=0, n_outside=1.0, transmittance=None,
                 offsets=None):
        """Create lens group.

        Args:
            positions: Element center positions (N, 3)
            rotations: Euler angles in degrees (N, 3)
            curvatures: Surface curvatures (N,)
            conics: Conic constants (N,)
            aspherics: Aspheric coefficients (N, K)
            radii: Aperture radii (N,)
            n_inside: Refractive index of material (N,) or scalar
            optical_stage: Stage in optical system (0=primary, etc.)
            n_outside: Ambient refractive index (default 1.0)
            transmittance: Bulk transmission (N,), default 1.0
            offsets: Surface offsets (N, 2), default zeros
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvatures = jnp.asarray(curvatures)
        self.conics = jnp.asarray(conics)
        self.aspherics = jnp.asarray(aspherics)
        self.radii = jnp.asarray(radii)
        self.optical_stage = int(optical_stage)

        n_elements = self.positions.shape[0]

        n_inside_arr = jnp.asarray(n_inside)
        if n_inside_arr.ndim == 0:
            n_inside_arr = jnp.full(n_elements, n_inside_arr)
        self.n_inside = n_inside_arr

        self.n_outside = float(n_outside)
        self.offsets = jnp.asarray(offsets) if offsets is not None else jnp.zeros((n_elements, 2))
        self.transmittance = jnp.asarray(transmittance) if transmittance is not None else jnp.ones(n_elements)

        # Determine if all elements are pure conics (no aspheric terms)
        self.is_pure_conic = bool(np.all(np.asarray(aspherics) == 0))

        self.aperture_samples = jnp.zeros((n_elements, 0, 2))
        self.perturbation_angles = jnp.zeros((n_elements, 0, 2))
        self.perturbation_scale = jnp.zeros(n_elements)
        self.perturbation_key = jax.random.key(0)

    @property
    def interaction(self) -> InteractionType:
        return InteractionType.REFRACT

    def get_surface(self, element_idx):
        from ..core import AsphericSurface
        return AsphericSurface(
            self.curvatures[element_idx],
            self.conics[element_idx],
            self.aspherics[element_idx],
            is_pure_conic=self.is_pure_conic,
        )

    def check_aperture(self, x, y, element_idx):
        return x**2 + y**2 <= self.radii[element_idx]**2

    def get_sampling_params(self):
        return {
            'type': 'disk',
            'radii': self.radii,
            'offsets': self.offsets,
        }

    def transform_to_world(self):
        """Transform to world coordinates.

        Returns:
            Tuple of (points, normals, weights, n_inside, transmittance)
        """
        points, normals, weights = _transform_disk_geometry_to_world(
            self.aperture_samples, self.offsets, self.curvatures, self.conics,
            self.aspherics, self.radii, self.perturbation_angles, self.positions,
            self.rotations, self.perturbation_scale
        )
        return points, normals, weights, self.n_inside, self.transmittance

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert lens element at index to config dict."""
        aspheric_list = [float(x) for x in np.asarray(self.aspherics[index])]
        # Remove trailing zeros from aspheric coefficients
        while aspheric_list and aspheric_list[-1] == 0.0:
            aspheric_list.pop()

        return {
            "type": "aspheric_disk",
            "position": [float(x) for x in np.asarray(self.positions[index])],
            "orientation": [float(x) for x in np.asarray(self.rotations[index])],
            "curvature": float(self.curvatures[index]),
            "conic": float(self.conics[index]),
            "aspheric": aspheric_list,
            "radius": float(self.radii[index]),
            "n_inside": float(self.n_inside[index]),
            "n_outside": float(self.n_outside),
            "transmittance": float(self.transmittance[index]),
            "offset": [float(x) for x in np.asarray(self.offsets[index])],
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]], **kwargs: Any) -> "AsphericDiskLensGroup":
        """Create AsphericDiskLensGroup from config dicts."""
        optical_stage = kwargs.get("optical_stage", 0)

        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]
        curvatures = [c["curvature"] for c in configs]
        conics = [c["conic"] for c in configs]
        radii = [c["radius"] for c in configs]
        n_inside = [c["n_inside"] for c in configs]
        transmittance = [c.get("transmittance", 1.0) for c in configs]
        offsets = [c.get("offset", [0.0, 0.0]) for c in configs]

        # Pad aspheric coefficients to same length
        aspheric_lists = [c.get("aspheric", []) for c in configs]
        max_len = max((len(a) for a in aspheric_lists), default=0)
        if max_len == 0:
            max_len = 1  # Ensure at least one element
        aspherics = [a + [0.0] * (max_len - len(a)) for a in aspheric_lists]

        # Get n_outside from first config (assumed same for all)
        n_outside = configs[0].get("n_outside", 1.0) if configs else 1.0

        return cls(
            positions=positions,
            rotations=rotations,
            curvatures=curvatures,
            conics=conics,
            aspherics=aspherics,
            radii=radii,
            n_inside=n_inside,
            optical_stage=optical_stage,
            n_outside=n_outside,
            transmittance=transmittance,
            offsets=offsets,
        )


class PlanoSlabGroup(LensGroup):
    """Group of flat parallel-sided windows (slabs).

    Rays refract at entry, propagate through material, and refract at exit.
    For parallel surfaces, exiting ray direction equals entering direction.

    Use case: camera entrance windows, protective covers, filters.
    """

    config_type: ClassVar[str] = "plano_slab"

    positions: jax.Array      # (N, 3)
    rotations: jax.Array      # (N, 3)
    curvatures: jax.Array     # (N,) all zeros
    conics: jax.Array         # (N,) all zeros
    aspherics: jax.Array      # (N, 1) all zeros
    offsets: jax.Array        # (N, 2) all zeros
    radii: jax.Array          # (N,)
    thickness: jax.Array      # (N,)
    n_inside: jax.Array       # (N,)
    transmittance: jax.Array  # (N,)
    aperture_samples: jax.Array   # (N, M, 2)
    perturbation_angles: jax.Array  # (N, M, 2)
    perturbation_scale: jax.Array
    perturbation_key: jax.Array

    n_outside: float
    optical_stage: int = eqx.field(static=True)
    is_pure_conic: bool = eqx.field(static=True)  # Always True for plano slabs

    def __init__(self, positions, rotations, radii, thickness, n_inside,
                 optical_stage=0, n_outside=1.0, transmittance=None):
        """Create flat window group.

        Args:
            positions: Window center positions (N, 3)
            rotations: Euler angles in degrees (N, 3)
            radii: Aperture radii (N,)
            thickness: Window thickness (N,) or scalar
            n_inside: Refractive index (N,) or scalar
            optical_stage: Stage in optical system
            n_outside: Ambient refractive index (default 1.0)
            transmittance: Bulk transmission (N,), default 1.0
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.radii = jnp.asarray(radii)
        self.optical_stage = int(optical_stage)

        n_elements = self.positions.shape[0]

        self.curvatures = jnp.zeros(n_elements)
        self.conics = jnp.zeros(n_elements)
        self.aspherics = jnp.zeros((n_elements, 1))
        self.offsets = jnp.zeros((n_elements, 2))

        # Plano slabs are always pure conics (flat surfaces)
        self.is_pure_conic = True

        thickness_arr = jnp.asarray(thickness)
        if thickness_arr.ndim == 0:
            thickness_arr = jnp.full(n_elements, thickness_arr)
        self.thickness = thickness_arr

        n_inside_arr = jnp.asarray(n_inside)
        if n_inside_arr.ndim == 0:
            n_inside_arr = jnp.full(n_elements, n_inside_arr)
        self.n_inside = n_inside_arr

        self.n_outside = float(n_outside)
        self.transmittance = jnp.asarray(transmittance) if transmittance is not None else jnp.ones(n_elements)

        self.aperture_samples = jnp.zeros((n_elements, 0, 2))
        self.perturbation_angles = jnp.zeros((n_elements, 0, 2))
        self.perturbation_scale = jnp.zeros(n_elements)
        self.perturbation_key = jax.random.key(0)

    @property
    def interaction(self) -> InteractionType:
        return InteractionType.SLAB

    def get_surface(self, element_idx):
        from ..core import AsphericSurface
        return AsphericSurface(
            self.curvatures[element_idx],
            self.conics[element_idx],
            self.aspherics[element_idx],
            is_pure_conic=self.is_pure_conic,
        )

    def check_aperture(self, x, y, element_idx):
        return x**2 + y**2 <= self.radii[element_idx]**2

    def get_sampling_params(self):
        return {
            'type': 'disk',
            'radii': self.radii,
            'offsets': self.offsets,
        }

    def transform_to_world(self):
        """Transform to world coordinates.

        Returns:
            Tuple of (points, normals, weights, n_inside, transmittance, thickness)
        """
        points, normals, weights = _transform_disk_geometry_to_world(
            self.aperture_samples, self.offsets, self.curvatures, self.conics,
            self.aspherics, self.radii, self.perturbation_angles, self.positions,
            self.rotations, self.perturbation_scale
        )
        return points, normals, weights, self.n_inside, self.transmittance, self.thickness

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert slab element at index to config dict."""
        return {
            "type": "plano_slab",
            "position": [float(x) for x in np.asarray(self.positions[index])],
            "orientation": [float(x) for x in np.asarray(self.rotations[index])],
            "radius": float(self.radii[index]),
            "thickness": float(self.thickness[index]),
            "n_inside": float(self.n_inside[index]),
            "n_outside": float(self.n_outside),
            "transmittance": float(self.transmittance[index]),
        }

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]], **kwargs: Any) -> "PlanoSlabGroup":
        """Create PlanoSlabGroup from config dicts."""
        optical_stage = kwargs.get("optical_stage", 0)

        positions = [c["position"] for c in configs]
        rotations = [c["orientation"] for c in configs]
        radii = [c["radius"] for c in configs]
        thickness = [c["thickness"] for c in configs]
        n_inside = [c["n_inside"] for c in configs]
        transmittance = [c.get("transmittance", 1.0) for c in configs]

        # Get n_outside from first config (assumed same for all)
        n_outside = configs[0].get("n_outside", 1.0) if configs else 1.0

        return cls(
            positions=positions,
            rotations=rotations,
            radii=radii,
            thickness=thickness,
            n_inside=n_inside,
            optical_stage=optical_stage,
            n_outside=n_outside,
            transmittance=transmittance,
        )
