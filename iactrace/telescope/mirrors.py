import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

from ..core import euler_to_matrix
from ..core import AsphericSurface
from ..core import DiskAperture, PolygonAperture

class Mirror(eqx.Module):
    """Single mirror element with surface, aperture, and sampled geometry."""
    
    position: jax.Array      # (3,)
    rotation: jax.Array      # (3,) euler angles in degrees
    surface: eqx.Module      # AsphericSurface
    aperture: eqx.Module     # DiskAperture, PolygonAperture
    offset: jax.Array        # (2,) x0, y0 offset on parent surface
    
    # Sampled geometry in local coordinates
    points: jax.Array        # (M, 3)
    normals: jax.Array       # (M, 3)
    weights: jax.Array       # (M, 1)
    
    optical_stage: int = eqx.field(static=True)  # 0=primary, 1=secondary, etc.
    
    def __init__(self, position, rotation, surface, aperture,
                 points=None, normals=None, weights=None, optical_stage=0, offset=None):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.surface = surface
        self.aperture = aperture
        self.offset = jnp.asarray(offset) if offset is not None else jnp.zeros(2)
        self.points = points if points is not None else jnp.zeros((0, 3))
        self.normals = normals if normals is not None else jnp.zeros((0, 3))
        self.weights = weights if weights is not None else jnp.zeros((0, 1))
        self.optical_stage = int(optical_stage)
    
    def sample(self, n_samples, key):
        """Return new Mirror with sampled surface points and normals."""
        xy = self.aperture.sample(key, (n_samples,))
        points, normals = self.surface.point_and_normal(xy, self.offset)
        
        # Weight = cos(angle to z-axis) normalized by area
        cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
        weights = cos_z / self.aperture.area() * n_samples
        
        return eqx.tree_at(
            lambda m: (m.points, m.normals, m.weights),
            self,
            (points, normals, weights)
        )
    
    def transform_to_world(self):
        """Return world-space points, normals, and weights."""
        rot = euler_to_matrix(self.rotation)
        points_world = jnp.einsum('ij,nj->ni', rot, self.points) + self.position
        normals_world = jnp.einsum('ij,nj->ni', rot, self.normals)
        return points_world, normals_world, self.weights


class MirrorGroup(eqx.Module):
    """Base class for grouped mirrors with shared surface type and aperture type."""

    # Transformations (one per mirror)
    positions: jax.Array      # (N, 3)
    rotations: jax.Array      # (N, 3) euler angles in degrees

    # Sampled geometry in local coordinates
    points: jax.Array         # (N, M, 3) - N mirrors, M samples each
    normals: jax.Array        # (N, M, 3)
    weights: jax.Array        # (N, M, 1)
    
    optical_stage: int = eqx.field(static=True)  # 0=primary, 1=secondary, etc.

    @abstractmethod
    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        pass

    @abstractmethod
    def transform_to_world(self):
        """Batch transform all mirrors to world coordinates."""
        pass
    
    @abstractmethod
    def get_surface(self):
        """Return the surface object for intersection calculations."""
        pass

    @abstractmethod
    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within aperture of specified mirror."""
        pass

    def __len__(self):
        """Return number of mirrors in group."""
        return self.positions.shape[0]


class AsphericDiskMirrorGroup(MirrorGroup):
    """Group of mirrors with aspheric surfaces and circular apertures."""

    # Shared surface parameters
    curvature: float
    conic: float
    aspheric: jax.Array       # (K,) polynomial coefficients

    # Per-mirror aperture data
    radii: jax.Array          # (N,) - disk radius for each mirror
    offsets: jax.Array        # (N, 2) - x0, y0 offset for each mirror

    def __init__(self, positions, rotations, surface, radii, optical_stage=0, offsets=None):
        """
        Create group from positions, rotations, shared surface, and per-mirror radii.

        Args:
            positions: (N, 3) array of mirror positions
            rotations: (N, 3) array of euler angles in degrees
            surface: AsphericSurface shared by all mirrors
            radii: (N,) array of disk radii
            optical_stage: Optical stage index (0=primary, 1=secondary, ...)
            offsets: (N, 2) array of x0, y0 offsets, or None for zeros
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvature = surface.curvature
        self.conic = surface.conic
        self.aspheric = surface.aspheric
        self.radii = jnp.asarray(radii)
        self.optical_stage = int(optical_stage)
        
        n_mirrors = self.positions.shape[0]
        self.offsets = jnp.asarray(offsets) if offsets is not None else jnp.zeros((n_mirrors, 2))

        # Initialize empty sampled geometry
        self.points = jnp.zeros((n_mirrors, 0, 3))
        self.normals = jnp.zeros((n_mirrors, 0, 3))
        self.weights = jnp.zeros((n_mirrors, 0, 1))

    def get_surface(self):
        """Return the surface object for intersection calculations."""
        return AsphericSurface(self.curvature, self.conic, self.aspheric)

    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        from ..utils.sampling import sample_disk

        n_mirrors = len(self)
        surface = self.get_surface()

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, radius, offset):
            # Sample aperture
            pts = sample_disk(key, (n_samples_per_mirror,))
            xy = pts * radius

            # Get surface geometry with offset
            points, normals = surface.point_and_normal(xy, offset)

            # Compute weights
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            area = jnp.pi * radius**2
            weights = cos_z / area * n_samples_per_mirror

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(keys, self.radii, self.offsets)

        return eqx.tree_at(
            lambda g: (g.points, g.normals, g.weights),
            self,
            (points, normals, weights)
        )

    def transform_to_world(self):
        """Batch transform all mirrors to world coordinates."""
        def transform_single(points, normals, weights, position, rotation):
            rot = euler_to_matrix(rotation)
            points_world = jnp.einsum('ij,nj->ni', rot, points) + position
            normals_world = jnp.einsum('ij,nj->ni', rot, normals)
            return points_world, normals_world, weights

        # Vmap over all mirrors
        return jax.vmap(transform_single)(
            self.points, self.normals, self.weights,
            self.positions, self.rotations
        )

    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within mirror aperture."""
        return x**2 + y**2 <= self.radii[mirror_idx]**2


class AsphericPolygonMirrorGroup(MirrorGroup):
    """Group of mirrors with aspheric surfaces and polygon apertures (same vertex count)."""

    # Shared surface parameters
    curvature: float
    conic: float
    aspheric: jax.Array       # (K,) polynomial coefficients

    # Per-mirror aperture data
    vertices: jax.Array       # (N, K, 2) - K vertices for each of N mirrors
    n_vertices: int           # Number of vertices (3, 4, 6, etc.)
    offsets: jax.Array        # (N, 2) - x0, y0 offset for each mirror

    def __init__(self, positions, rotations, surface, vertices_list, optical_stage=0, offsets=None):
        """
        Create group from positions, rotations, shared surface, and per-mirror vertices.

        Args:
            positions: (N, 3) array of mirror positions
            rotations: (N, 3) array of euler angles in degrees
            surface: AsphericSurface shared by all mirrors
            vertices_list: (N, K, 2) array of polygon vertices
            optical_stage: Optical stage index (0=primary, 1=secondary, ...)
            offsets: (N, 2) array of x0, y0 offsets, or None for zeros
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvature = surface.curvature
        self.conic = surface.conic
        self.aspheric = surface.aspheric
        self.vertices = jnp.asarray(vertices_list)
        self.n_vertices = self.vertices.shape[1]
        self.optical_stage = int(optical_stage)
        
        n_mirrors = self.positions.shape[0]
        self.offsets = jnp.asarray(offsets) if offsets is not None else jnp.zeros((n_mirrors, 2))

        # Initialize empty sampled geometry
        self.points = jnp.zeros((n_mirrors, 0, 3))
        self.normals = jnp.zeros((n_mirrors, 0, 3))
        self.weights = jnp.zeros((n_mirrors, 0, 1))

    def get_surface(self):
        """Return the surface object for intersection calculations."""
        return AsphericSurface(self.curvature, self.conic, self.aspheric)

    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        from ..utils.sampling import sample_polygon

        n_mirrors = len(self)
        surface = self.get_surface()

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, vertices, offset):
            # Sample aperture
            xy = sample_polygon(key, vertices, (n_samples_per_mirror,))

            # Get surface geometry with offset
            points, normals = surface.point_and_normal(xy, offset)

            # Compute weights
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            # Polygon area using shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
            weights = cos_z / area * n_samples_per_mirror

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(keys, self.vertices, self.offsets)

        return eqx.tree_at(
            lambda g: (g.points, g.normals, g.weights),
            self,
            (points, normals, weights)
        )

    def transform_to_world(self):
        """Batch transform all mirrors to world coordinates."""
        def transform_single(points, normals, weights, position, rotation):
            rot = euler_to_matrix(rotation)
            points_world = jnp.einsum('ij,nj->ni', rot, points) + position
            normals_world = jnp.einsum('ij,nj->ni', rot, normals)
            return points_world, normals_world, weights

        # Vmap over all mirrors
        return jax.vmap(transform_single)(
            self.points, self.normals, self.weights,
            self.positions, self.rotations
        )

    def check_aperture(self, x, y, mirror_idx):
        """Check if points (x, y) are within mirror aperture (convex polygon)."""
        verts = self.vertices[mirror_idx]
        n = self.n_vertices
        
        def edge_check(carry, i):
            v1, v2 = verts[i], verts[(i + 1) % n]
            cross = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0])
            return carry & (cross >= 0), None
        
        inside, _ = jax.lax.scan(edge_check, jnp.ones_like(x, dtype=bool), jnp.arange(n))
        return inside


def group_mirrors(mirrors):
    """
    Convert list of mirrors to list of groups by optical stage, surface type, 
    aperture type, and vertex count.

    Groups mirrors that share:
    - Same optical stage
    - Same surface type (AsphericSurface) with identical parameters
    - Same aperture type (Disk or Polygon)
    - For polygons: same number of vertices

    Args:
        mirrors: List of Mirror objects

    Returns:
        List of MirrorGroup objects (AsphericDiskMirrorGroup or AsphericPolygonMirrorGroup)
    """
    if not mirrors:
        return []

    from collections import defaultdict
    
    groups = []

    # First, separate by optical stage
    by_stage = defaultdict(list)
    for m in mirrors:
        by_stage[m.optical_stage].append(m)

    for stage, stage_mirrors in sorted(by_stage.items()):
        # Group aspheric mirrors with disk apertures
        disk_mirrors = [m for m in stage_mirrors
                        if isinstance(m.surface, AsphericSurface)
                        and isinstance(m.aperture, DiskAperture)]

        # Further group by matching surface parameters
        disk_by_surface = _group_by_surface_params(disk_mirrors)

        for surface_key, mirror_list in disk_by_surface.items():
            if not mirror_list:
                continue

            positions = jnp.stack([m.position for m in mirror_list])
            rotations = jnp.stack([m.rotation for m in mirror_list])
            radii = jnp.array([m.aperture.radius for m in mirror_list])
            offsets = jnp.stack([m.offset for m in mirror_list])
            surface = mirror_list[0].surface  # All share same surface

            groups.append(AsphericDiskMirrorGroup(
                positions, rotations, surface, radii, optical_stage=stage, offsets=offsets
            ))

        # Group aspheric mirrors with polygon apertures by vertex count
        poly_mirrors = [m for m in stage_mirrors
                        if isinstance(m.surface, AsphericSurface)
                        and isinstance(m.aperture, PolygonAperture)]

        # Group by vertex count first
        poly_by_nverts = defaultdict(list)
        for m in poly_mirrors:
            n_verts = len(m.aperture.vertices)
            poly_by_nverts[n_verts].append(m)

        # Then group by surface parameters within each vertex count
        for n_verts, mirror_list in poly_by_nverts.items():
            poly_by_surface = _group_by_surface_params(mirror_list)

            for surface_key, mlist in poly_by_surface.items():
                if not mlist:
                    continue

                positions = jnp.stack([m.position for m in mlist])
                rotations = jnp.stack([m.rotation for m in mlist])
                vertices_list = jnp.stack([m.aperture.vertices for m in mlist])
                offsets = jnp.stack([m.offset for m in mlist])
                surface = mlist[0].surface  # All share same surface

                groups.append(AsphericPolygonMirrorGroup(
                    positions, rotations, surface, vertices_list, optical_stage=stage, offsets=offsets
                ))

    return groups


def _group_by_surface_params(mirrors):
    """
    Group mirrors by identical surface parameters.

    Returns dict mapping surface parameter tuple to list of mirrors.
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for mirror in mirrors:
        if isinstance(mirror.surface, AsphericSurface):
            # Create hashable key from surface parameters
            curvature = mirror.surface.curvature
            conic = mirror.surface.conic
            aspheric_tuple = tuple(mirror.surface.aspheric.tolist())

            key = (curvature, conic, aspheric_tuple)
            grouped[key].append(mirror)

    return grouped