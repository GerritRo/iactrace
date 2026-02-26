from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.optics import InteractionType, OpticalGroupBase
from ..core.transforms import euler_to_matrix


def _point_in_convex_polygon(x, y, vertices, n_vertices):
    """Check if points (x, y) are inside a convex polygon.

    Assumes vertices are in counter-clockwise order (ensured at load time).
    For CCW vertices, a point is inside if all edge cross products are >= 0.

    Args:
        x: x-coordinates (can be scalar or array)
        y: y-coordinates (can be scalar or array)
        vertices: Polygon vertices in CCW order (K, 2)
        n_vertices: Number of vertices

    Returns:
        Boolean mask, True if inside polygon
    """
    def edge_check(carry, i):
        v1, v2 = vertices[i], vertices[(i + 1) % n_vertices]
        cross = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0])
        return carry & (cross >= 0), None

    inside, _ = jax.lax.scan(edge_check, jnp.ones_like(x, dtype=bool), jnp.arange(n_vertices))
    return inside


def _polygon_area(vertices):
    """Compute area of convex polygon using shoelace formula.

    Args:
        vertices: Polygon vertices (K, 2)

    Returns:
        Polygon area (scalar)
    """
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    return 0.5 * jnp.abs(jnp.sum(vx * jnp.roll(vy, -1) - jnp.roll(vx, -1) * vy))


def _transform_to_world_common(aperture_samples, offsets, curvatures, conics, aspherics,
                                aperture_data, perturbation_angles, positions, rotations,
                                perturbation_scale, reflectivity, area_fn):
    """Common transform_to_world logic for all mirror group types.

    Args:
        aperture_samples: 2D sample positions on aperture (N, M, 2)
        offsets: Per-mirror offsets (N, 2)
        curvatures: Per-mirror curvatures (N,)
        conics: Per-mirror conic constants (N,)
        aspherics: Per-mirror aspheric coefficients (N, K)
        aperture_data: Aperture-specific data (radii for disk, vertices for polygon)
        perturbation_angles: Random angles for roughness (N, M, 2)
        positions: Mirror positions (N, 3)
        rotations: Mirror rotations (N, 3)
        perturbation_scale: Roughness scale per mirror (N,)
        reflectivity: Reflectivity per mirror (N,)
        area_fn: Function to compute aperture area from aperture_data

    Returns:
        Tuple of (points_world, normals_world, weights)
    """
    from ..core.optics import apply_perturbation
    from ..core.surfaces import compute_sag_and_normal

    def compute_and_transform_single(xy, offset, curvature, conic, aspheric, ap_data,
                                      angles, position, rotation, scale, refl):
        # Compute sag and normals for all sample points
        x, y = xy[..., 0], xy[..., 1]
        points, normals = jax.vmap(
            lambda xi, yi: compute_sag_and_normal(xi, yi, offset, curvature, conic, aspheric)
        )(x, y)

        # Compute weights: cos(angle to z-axis) / area * n_samples / reflectivity
        cos_z = jnp.sum(normals * jnp.array([0.0, 0.0, 1.0]), axis=-1, keepdims=True)
        n_samples = xy.shape[0]
        area = area_fn(ap_data)
        weights = cos_z / area * n_samples / refl

        # Transform to world coordinates
        rot = euler_to_matrix(rotation)
        points_world = jnp.einsum('ij,nj->ni', rot, points) + position
        normals_world = jnp.einsum('ij,nj->ni', rot, normals)

        # Apply perturbation using current normals and stored random angles
        perturbed = apply_perturbation(normals_world, angles, scale)

        return points_world, perturbed, weights

    return jax.vmap(compute_and_transform_single)(
        aperture_samples, offsets, curvatures, conics, aspherics,
        aperture_data, perturbation_angles, positions, rotations,
        perturbation_scale, reflectivity
    )


class MirrorGroup(OpticalGroupBase):
    """Base class for grouped mirrors with shared surface type and aperture type."""

    config_type: ClassVar[str]

    # Transformations (one per mirror)
    positions: jax.Array      # (N, 3)
    rotations: jax.Array      # (N, 3) euler angles in degrees

    # 2D sample positions on aperture
    aperture_samples: jax.Array   # (N, M, 2) - 2D aperture coordinates

    # Perturbation: random angles in tangent space (recomputed at render time)
    perturbation_angles: jax.Array  # (N, M, 2) - (theta1, theta2) random values
    perturbation_scale: jax.Array   # (N,) - per-mirror scale factor (radians)
    perturbation_key: jax.Array     # PRNGKey for deterministic perturbation in ray tracing

    # Per-mirror reflectivity scale (divides computed weights)
    reflectivity: jax.Array         # (N,) - reflectivity for each mirror (default 1.0)

    optical_stage: int = eqx.field(static=True)  # 0=primary, 1=secondary, etc.

    @property
    def interaction(self) -> InteractionType:
        """Mirrors always reflect."""
        return InteractionType.REFLECT

    @abstractmethod
    def get_surface(self, mirror_idx):
        """Return the surface object for a specific mirror."""
        ...

    @abstractmethod
    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within aperture of specified mirror."""
        ...

    @abstractmethod
    def get_sampling_params(self):
        """Return structured dict with geometry parameters for sampling."""
        ...

    @abstractmethod
    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates.

        Computes points/normals/weights dynamically from aperture_samples and current
        surface parameters (curvature, conic, aspheric), then transforms to world
        coordinates with perturbation applied.

        Returns:
            Tuple of (points_world, normals_world, weights) arrays
        """
        ...

    @abstractmethod
    def to_config(self, index: int) -> dict[str, Any]:
        """Convert a single mirror at index to a config dict.

        Args:
            index: Index of the mirror within the group.

        Returns:
            Configuration dictionary for this mirror (without template).
        """
        ...

    @abstractmethod
    def get_surface_params(self, index: int) -> dict[str, Any]:
        """Get surface parameters for a single mirror.

        Args:
            index: Index of the mirror within the group.

        Returns:
            Dictionary with curvature, conic, aspheric coefficients.
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        templates: dict[str, Any],
        optical_stage: int = 0,
    ) -> MirrorGroup:
        """Create a MirrorGroup from a list of config dicts.

        Args:
            configs: List of mirror configurations with aperture, position, etc.
            templates: Dictionary of surface templates.
            optical_stage: Optical stage (0=primary, 1=secondary, etc.).

        Returns:
            New MirrorGroup instance.
        """
        ...

    def __len__(self):
        """Return number of mirrors in group."""
        return self.positions.shape[0]


class AsphericDiskMirrorGroup(MirrorGroup):
    """Group of mirrors with aspheric surfaces and circular/annular apertures.

    Each mirror has individual surface parameters (curvature, conic, aspherics).
    Supports optional center hole via inner_radii parameter.
    Geometry is computed dynamically from aperture_samples and current surface parameters.
    """

    config_type: ClassVar[str] = "circular"

    # Per-mirror surface parameters
    curvatures: jax.Array     # (N,) curvature for each mirror
    conics: jax.Array         # (N,) conic constant for each mirror
    aspherics: jax.Array      # (N, K_max) polynomial coefficients, padded

    # Per-mirror aperture data
    radii: jax.Array          # (N,) - disk radius for each mirror
    inner_radii: jax.Array    # (N,) - inner radius (hole) for each mirror, 0 for solid disk
    offsets: jax.Array        # (N, 2) - x0, y0 offset for each mirror

    is_pure_conic: bool = eqx.field(static=True)  # True if all aspherics are zero

    def __init__(self, positions, rotations, curvatures, conics, aspherics, radii,
                 optical_stage=0, offsets=None, inner_radii=None):
        """
        Create group from positions, rotations, per-mirror surface params, and radii.
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvatures = jnp.asarray(curvatures)
        self.conics = jnp.asarray(conics)
        self.aspherics = jnp.asarray(aspherics)
        self.radii = jnp.asarray(radii)
        self.optical_stage = int(optical_stage)

        n_mirrors = self.positions.shape[0]
        self.offsets = jnp.asarray(offsets) if offsets is not None else jnp.zeros((n_mirrors, 2))
        self.inner_radii = jnp.asarray(inner_radii) if inner_radii is not None else jnp.zeros(n_mirrors)

        # Determine if all mirrors are pure conics (no aspheric terms)
        self.is_pure_conic = bool(np.all(np.asarray(aspherics) == 0))

        # Initialize empty - will be set by integrator
        self.aperture_samples = jnp.zeros((n_mirrors, 0, 2))
        self.perturbation_angles = jnp.zeros((n_mirrors, 0, 2))
        self.perturbation_scale = jnp.zeros(n_mirrors)
        self.perturbation_key = jax.random.key(0)
        self.reflectivity = jnp.ones(n_mirrors)

    def get_surface(self, mirror_idx):
        """Return the surface object for a specific mirror."""
        from ..core import AsphericSurface
        return AsphericSurface(
            self.curvatures[mirror_idx],
            self.conics[mirror_idx],
            self.aspherics[mirror_idx],
            is_pure_conic=self.is_pure_conic,
        )

    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within mirror aperture (between inner and outer radius)."""
        r_sq = x**2 + y**2
        return (r_sq >= self.inner_radii[mirror_idx]**2) & (r_sq <= self.radii[mirror_idx]**2)

    def get_sampling_params(self):
        """Return structured dict with geometry parameters for sampling."""
        return {
            'type': 'disk',
            'radii': self.radii,
            'inner_radii': self.inner_radii,
            'offsets': self.offsets,
        }

    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates."""
        radii_stacked = jnp.stack([self.inner_radii, self.radii], axis=-1)
        return _transform_to_world_common(
            self.aperture_samples, self.offsets, self.curvatures, self.conics, self.aspherics,
            radii_stacked, self.perturbation_angles, self.positions, self.rotations,
            self.perturbation_scale, self.reflectivity,
            area_fn=lambda r: jnp.pi * (r[1]**2 - r[0]**2)
        )

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert mirror at index to config dict."""
        offset = [float(o) for o in np.asarray(self.offsets[index])]
        config: dict[str, Any] = {
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "aperture": {
                "type": "circular",
                "radius": float(self.radii[index]),
                "inner_radius": float(self.inner_radii[index]),
            },
        }
        if self.optical_stage != 0:
            config["stage"] = self.optical_stage
        if not (offset[0] == 0.0 and offset[1] == 0.0):
            config["offset"] = offset
        return config

    def get_surface_params(self, index: int) -> dict[str, Any]:
        """Get surface parameters for a single mirror."""
        aspheric = [float(a) for a in np.asarray(self.aspherics[index])]
        # Trim trailing zeros
        while aspheric and aspheric[-1] == 0.0:
            aspheric.pop()
        return {
            "curvature": float(self.curvatures[index]),
            "conic": float(self.conics[index]),
            "aspheric": aspheric,
        }

    @classmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        templates: dict[str, Any],
        optical_stage: int = 0,
    ) -> AsphericDiskMirrorGroup:
        """Create AsphericDiskMirrorGroup from config dicts."""
        positions = []
        rotations = []
        curvatures = []
        conics = []
        aspherics = []
        radii = []
        inner_radii = []
        offsets = []

        for config in configs:
            positions.append(config["position"])
            rotations.append(config["orientation"])
            radii.append(config["aperture"]["radius"])
            inner_radii.append(config["aperture"].get("inner_radius", 0))
            offsets.append(config.get("offset", [0.0, 0.0]))

            # Get surface params from template or config override
            template_name = config.get("template")
            if template_name and template_name in templates:
                surface = templates[template_name]["surface"]
                curvatures.append(config.get("curvature", surface["curvature"]))
                conics.append(config.get("conic", surface["conic"]))
                aspherics.append(config.get("aspheric", surface.get("aspheric", [])))
            else:
                curvatures.append(config["curvature"])
                conics.append(config["conic"])
                aspherics.append(config.get("aspheric", []))

        return cls(
            positions=positions,
            rotations=rotations,
            curvatures=curvatures,
            conics=conics,
            aspherics=_pad_aspherics(aspherics),
            radii=radii,
            inner_radii=inner_radii,
            optical_stage=optical_stage,
            offsets=offsets,
        )


class AsphericPolygonMirrorGroup(MirrorGroup):
    """Group of mirrors with aspheric surfaces and polygon apertures (same vertex count).

    Each mirror has individual surface parameters (curvature, conic, aspherics).
    Geometry is computed dynamically from aperture_samples and current surface parameters.
    """

    config_type: ClassVar[str] = "polygon"

    # Per-mirror surface parameters
    curvatures: jax.Array     # (N,) curvature for each mirror
    conics: jax.Array         # (N,) conic constant for each mirror
    aspherics: jax.Array      # (N, K_max) polynomial coefficients, padded

    # Per-mirror aperture data
    vertices: jax.Array       # (N, K, 2) - K vertices for each of N mirrors
    n_vertices: int = eqx.field(static=True)  # Number of vertices (3, 4, 6, etc.)
    offsets: jax.Array        # (N, 2) - x0, y0 offset for each mirror

    is_pure_conic: bool = eqx.field(static=True)  # True if all aspherics are zero

    def __init__(self, positions, rotations, curvatures, conics, aspherics, vertices_list,
                 optical_stage=0, offsets=None):
        """
        Create group from positions, rotations, per-mirror surface params, and vertices.
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvatures = jnp.asarray(curvatures)
        self.conics = jnp.asarray(conics)
        self.aspherics = jnp.asarray(aspherics)
        self.vertices = jnp.asarray(vertices_list)
        self.n_vertices = int(self.vertices.shape[1])
        self.optical_stage = int(optical_stage)

        n_mirrors = self.positions.shape[0]
        self.offsets = jnp.asarray(offsets) if offsets is not None else jnp.zeros((n_mirrors, 2))

        # Determine if all mirrors are pure conics (no aspheric terms)
        self.is_pure_conic = bool(np.all(np.asarray(aspherics) == 0))

        # Initialize empty - will be set by integrator
        self.aperture_samples = jnp.zeros((n_mirrors, 0, 2))
        self.perturbation_angles = jnp.zeros((n_mirrors, 0, 2))
        self.perturbation_scale = jnp.zeros(n_mirrors)
        self.perturbation_key = jax.random.key(0)
        self.reflectivity = jnp.ones(n_mirrors)

    def get_surface(self, mirror_idx):
        """Return the surface object for a specific mirror."""
        from ..core import AsphericSurface
        return AsphericSurface(
            self.curvatures[mirror_idx],
            self.conics[mirror_idx],
            self.aspherics[mirror_idx],
            is_pure_conic=self.is_pure_conic,
        )

    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within mirror aperture (convex polygon)."""
        return _point_in_convex_polygon(x, y, self.vertices[mirror_idx], self.n_vertices)

    def get_sampling_params(self):
        """Return structured dict with geometry parameters for sampling."""
        return {
            'type': 'polygon',
            'vertices': self.vertices,
            'offsets': self.offsets,
        }

    def transform_to_world(self):
        """Compute geometry from current surface params and transform to world coordinates."""
        return _transform_to_world_common(
            self.aperture_samples, self.offsets, self.curvatures, self.conics, self.aspherics,
            self.vertices, self.perturbation_angles, self.positions, self.rotations,
            self.perturbation_scale, self.reflectivity,
            area_fn=_polygon_area
        )

    def to_config(self, index: int) -> dict[str, Any]:
        """Convert mirror at index to config dict."""
        offset = [float(o) for o in np.asarray(self.offsets[index])]
        vertices = [[float(v[0]), float(v[1])] for v in np.asarray(self.vertices[index])]
        config: dict[str, Any] = {
            "position": [float(p) for p in np.asarray(self.positions[index])],
            "orientation": [float(r) for r in np.asarray(self.rotations[index])],
            "aperture": {
                "type": "polygon",
                "vertices": vertices,
            },
        }
        if self.optical_stage != 0:
            config["stage"] = self.optical_stage
        if not (offset[0] == 0.0 and offset[1] == 0.0):
            config["offset"] = offset
        return config

    def get_surface_params(self, index: int) -> dict[str, Any]:
        """Get surface parameters for a single mirror."""
        aspheric = [float(a) for a in np.asarray(self.aspherics[index])]
        # Trim trailing zeros
        while aspheric and aspheric[-1] == 0.0:
            aspheric.pop()
        return {
            "curvature": float(self.curvatures[index]),
            "conic": float(self.conics[index]),
            "aspheric": aspheric,
        }

    @classmethod
    def from_config(
        cls,
        configs: list[dict[str, Any]],
        templates: dict[str, Any],
        optical_stage: int = 0,
    ) -> AsphericPolygonMirrorGroup:
        """Create AsphericPolygonMirrorGroup from config dicts."""
        positions = []
        rotations = []
        curvatures = []
        conics = []
        aspherics = []
        vertices_list = []
        offsets = []

        for config in configs:
            positions.append(config["position"])
            rotations.append(config["orientation"])
            vertices_list.append(config["aperture"]["vertices"])
            offsets.append(config.get("offset", [0.0, 0.0]))

            # Get surface params from template or config override
            template_name = config.get("template")
            if template_name and template_name in templates:
                surface = templates[template_name]["surface"]
                curvatures.append(config.get("curvature", surface["curvature"]))
                conics.append(config.get("conic", surface["conic"]))
                aspherics.append(config.get("aspheric", surface.get("aspheric", [])))
            else:
                curvatures.append(config["curvature"])
                conics.append(config["conic"])
                aspherics.append(config.get("aspheric", []))

        return cls(
            positions=positions,
            rotations=rotations,
            curvatures=curvatures,
            conics=conics,
            aspherics=_pad_aspherics(aspherics),
            vertices_list=jnp.array(vertices_list),
            optical_stage=optical_stage,
            offsets=offsets,
        )


def _pad_aspherics(aspheric_list):
    """
    Pad aspheric coefficient arrays to uniform length.

    Args:
        aspheric_list: List of (K_i,) arrays with varying lengths

    Returns:
        (N, K_max) array padded with zeros
    """
    if not aspheric_list:
        return jnp.zeros((0, 0))

    max_len = max(len(a) for a in aspheric_list)
    # Ensure at least length 1 to avoid empty array issues
    max_len = max(max_len, 1)

    padded = []
    for a in aspheric_list:
        a = jnp.asarray(a)
        if len(a) < max_len:
            a = jnp.concatenate([a, jnp.zeros(max_len - len(a))])
        padded.append(a)

    return jnp.stack(padded)