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
    
    # Sampled geometry in local coordinates
    points: jax.Array        # (M, 3)
    normals: jax.Array       # (M, 3)
    weights: jax.Array       # (M, 1)
    
    def __init__(self, position, rotation, surface, aperture,
                 points=None, normals=None, weights=None):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.surface = surface
        self.aperture = aperture
        self.points = points if points is not None else jnp.zeros((0, 3))
        self.normals = normals if normals is not None else jnp.zeros((0, 3))
        self.weights = weights if weights is not None else jnp.zeros((0, 1))
    
    def sample(self, n_samples, key):
        """Return new Mirror with sampled surface points and normals."""
        xy = self.aperture.sample(key, (n_samples,))
        points, normals = self.surface.point_and_normal(xy)
        
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

    @abstractmethod
    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        pass

    @abstractmethod
    def transform_to_world(self):
        """Batch transform all mirrors to world coordinates."""
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

    def __init__(self, positions, rotations, surface, radii):
        """
        Create group from positions, rotations, shared surface, and per-mirror radii.

        Args:
            positions: (N, 3) array of mirror positions
            rotations: (N, 3) array of euler angles in degrees
            surface: AsphericSurface shared by all mirrors
            radii: (N,) array of disk radii
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvature = surface.curvature
        self.conic = surface.conic
        self.aspheric = surface.aspheric
        self.radii = jnp.asarray(radii)

        # Initialize empty sampled geometry
        n_mirrors = self.positions.shape[0]
        self.points = jnp.zeros((n_mirrors, 0, 3))
        self.normals = jnp.zeros((n_mirrors, 0, 3))
        self.weights = jnp.zeros((n_mirrors, 0, 1))

    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        from ..utils.sampling import sample_disk

        n_mirrors = len(self)
        surface = AsphericSurface(self.curvature, self.conic, self.aspheric)

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, radius):
            # Sample aperture
            pts = sample_disk(key, (n_samples_per_mirror,))
            xy = pts * radius

            # Get surface geometry
            points, normals = surface.point_and_normal(xy)

            # Compute weights
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            area = jnp.pi * radius**2
            weights = cos_z / area * n_samples_per_mirror

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(keys, self.radii)

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


class AsphericPolygonMirrorGroup(MirrorGroup):
    """Group of mirrors with aspheric surfaces and polygon apertures (same vertex count)."""

    # Shared surface parameters
    curvature: float
    conic: float
    aspheric: jax.Array       # (K,) polynomial coefficients

    # Per-mirror aperture data
    vertices: jax.Array       # (N, K, 2) - K vertices for each of N mirrors
    n_vertices: int           # Number of vertices (3, 4, 6, etc.)

    def __init__(self, positions, rotations, surface, vertices_list):
        """
        Create group from positions, rotations, shared surface, and per-mirror vertices.

        Args:
            positions: (N, 3) array of mirror positions
            rotations: (N, 3) array of euler angles in degrees
            surface: AsphericSurface shared by all mirrors
            vertices_list: (N, K, 2) array of polygon vertices
        """
        self.positions = jnp.asarray(positions)
        self.rotations = jnp.asarray(rotations)
        self.curvature = surface.curvature
        self.conic = surface.conic
        self.aspheric = surface.aspheric
        self.vertices = jnp.asarray(vertices_list)
        self.n_vertices = self.vertices.shape[1]

        # Initialize empty sampled geometry
        n_mirrors = self.positions.shape[0]
        self.points = jnp.zeros((n_mirrors, 0, 3))
        self.normals = jnp.zeros((n_mirrors, 0, 3))
        self.weights = jnp.zeros((n_mirrors, 0, 1))

    def sample(self, n_samples_per_mirror, key):
        """Sample all mirrors in group with batched operations."""
        from ..utils.sampling import sample_polygon

        n_mirrors = len(self)
        surface = AsphericSurface(self.curvature, self.conic, self.aspheric)

        # Split key for each mirror
        keys = jax.random.split(key, n_mirrors)

        def sample_single_mirror(key, vertices):
            # Sample aperture
            xy = sample_polygon(key, vertices, (n_samples_per_mirror,))

            # Get surface geometry
            points, normals = surface.point_and_normal(xy)

            # Compute weights
            cos_z = jnp.sum(normals * jnp.array([0., 0., 1.]), axis=-1, keepdims=True)
            # Polygon area using shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
            weights = cos_z / area * n_samples_per_mirror

            return points, normals, weights

        # Vmap over all mirrors
        points, normals, weights = jax.vmap(sample_single_mirror)(keys, self.vertices)

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


def group_mirrors(mirrors):
    """
    Convert list of mirrors to list of groups by surface type, aperture type, and vertex count.

    Groups mirrors that share:
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

    groups = []

    # Group aspheric mirrors with disk apertures
    disk_mirrors = [m for m in mirrors
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
        surface = mirror_list[0].surface  # All share same surface

        groups.append(AsphericDiskMirrorGroup(positions, rotations, surface, radii))

    # Group aspheric mirrors with polygon apertures by vertex count
    poly_mirrors = [m for m in mirrors
                    if isinstance(m.surface, AsphericSurface)
                    and isinstance(m.aperture, PolygonAperture)]

    # Group by vertex count first
    from collections import defaultdict
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
            surface = mlist[0].surface  # All share same surface

            groups.append(AsphericPolygonMirrorGroup(positions, rotations, surface, vertices_list))

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