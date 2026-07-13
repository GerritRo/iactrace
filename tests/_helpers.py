"""Shared builders for the test suite.

Plain functions (not fixtures) so any test module can import exactly what it
needs. Fixtures live in ``conftest.py``; these are the parametrisable builders
that several modules would otherwise construct by copy-paste.
"""

import jax
import jax.numpy as jnp

from iactrace import Camera, SquareSensorGroup, Telescope
from iactrace.camera._hexgeom import SQRT3
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.obstructions import CylinderGroup
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import AsphericSurfaceGroup

# --- hexagonal sensor geometry -------------------------------------------------


def make_hex_centers(n_rings=2, hex_size=0.001):
    """Axial-coordinate centres of a hexagonal pixel grid."""
    centers = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(-n_rings, n_rings + 1):
            if max(abs(q), abs(r), abs(-q - r)) <= n_rings:
                x = hex_size * SQRT3 * (q + r / 2)
                y = hex_size * 1.5 * r
                centers.append([x, y])
    return jnp.array(centers)


# --- mirror / surface group builders ------------------------------------------


def make_disk_mirror_group(
    positions, rotations, curvatures, conics, aspherics, radii, optical_stage=0, n_samples=100
):
    """An OpticalElementGroup configured as a reflective aspheric disk mirror."""
    n = curvatures.shape[0]
    surface = AsphericSurfaceGroup(
        curvatures=curvatures,
        conics=conics,
        aspherics=aspherics,
        offsets=jnp.zeros((n, 2)),
    )
    aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=positions,
        rotations=rotations,
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=optical_stage,
        n_samples=n_samples,
    )


def mirror_group_with_surface(surface, radius=0.5, stage=0, n_samples=64):
    """Wrap an arbitrary surface group in a single disk-mirror element group.

    ``radius`` may be a scalar (shared aperture) or a per-element array.
    """
    n = surface.offsets.shape[0]
    radii = jnp.full(n, radius) if jnp.ndim(radius) == 0 else jnp.asarray(radius)
    aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=stage,
        n_samples=n_samples,
    )


def spherical_cap_surface(radius, sag):
    """Single-element spherical-cap AsphericSurfaceGroup for PMT photocathode tests.

    Mirrors the sag-to-curvature derivation the old ``PMT(face_sag=...)``
    shorthand used internally, back when the PMT stored a bespoke
    (radius, sag) pair instead of a general :class:`SurfaceGroup`. Pass the
    result as ``PMT(..., surface=spherical_cap_surface(r, h), vertex_z=h)``.
    """
    r_curv = (radius * radius + sag * sag) / (2.0 * sag)
    return AsphericSurfaceGroup(
        offsets=jnp.zeros((1, 2)),
        curvatures=jnp.array([-1.0 / r_curv]),
        conics=jnp.zeros(1),
        aspherics=jnp.zeros((1, 0)),
    )


# --- whole telescope + camera builders ----------------------------------------


def make_simple_telescope(curvature=1.0, n_samples=1024, key=None):
    """A minimal single-paraboloid telescope + square camera at its focus."""
    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([curvature])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = make_disk_mirror_group(
        positions,
        rotations,
        curvatures,
        conics,
        aspherics,
        radii,
        optical_stage=0,
        n_samples=n_samples,
    )

    focal_length = 1.0 / (2.0 * curvature) if curvature != 0 else 1000.0

    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )

    telescope = Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=None,
        name="test_telescope",
        camera_position=[0.0, 0.0, focal_length],
    )
    return telescope, Camera(sensor_groups=[sensor])


def make_two_stage_telescope(n_samples=512, key=None):
    """A two-stage (folded) telescope + camera."""
    first = make_disk_mirror_group(
        positions=jnp.array([[0.0, 0.0, 0.0]]),
        rotations=jnp.array([[0.0, 0.0, 0.0]]),
        curvatures=jnp.array([0.0]),
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.1]),
        optical_stage=0,
        n_samples=n_samples,
    )
    second = make_disk_mirror_group(
        positions=jnp.array([[0.0, 0.0, 0.5]]),
        rotations=jnp.array([[0.0, 45.0, 0.0]]),
        curvatures=jnp.array([0.0]),
        conics=jnp.array([0.0]),
        aspherics=jnp.zeros((1, 1)),
        radii=jnp.array([0.2]),
        optical_stage=1,
        n_samples=n_samples,
    )
    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=50,
        height=50,
        bounds=(-0.2, 0.2, -0.2, 0.2),
    )
    telescope = Telescope(
        mirror_groups=[first, second],
        obstruction_groups=None,
        name="two_stage_telescope",
        camera_position=[0.5, 0.0, 0.5],
        camera_rotation=[0.0, 90.0, 0.0],
    )
    return telescope, Camera(sensor_groups=[sensor])


def make_telescope_with_obstruction(n_samples=1024, key=None):
    """A single-paraboloid telescope + camera with a central obstruction."""
    positions = jnp.array([[0.0, 0.0, 0.0]])
    rotations = jnp.array([[0.0, 0.0, 0.0]])
    curvatures = jnp.array([1.0])
    conics = jnp.array([-1.0])
    aspherics = jnp.zeros((1, 1))
    radii = jnp.array([0.1])

    mirror_group = make_disk_mirror_group(
        positions,
        rotations,
        curvatures,
        conics,
        aspherics,
        radii,
        optical_stage=0,
        n_samples=n_samples,
    )
    obstruction = CylinderGroup(p1=[[0.0, 0.0, 0.05]], p2=[[0.0, 0.0, 0.2]], r=[0.03])
    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=100,
        height=100,
        bounds=(-0.018, 0.018, -0.018, 0.018),
    )
    telescope = Telescope(
        mirror_groups=[mirror_group],
        obstruction_groups=[obstruction],
        name="obstructed_telescope",
        camera_position=[0.0, 0.0, 0.5],
    )
    return telescope, Camera(sensor_groups=[sensor])


# --- sensor binning -----------------------------------------------------------


def bin_positions(sensor, sensor_idx, x, y, values):
    """Assign ``(x, y)`` to pixels and scatter *values* into the accumulator.

    The two-step the camera pipeline runs (``pixel_index_and_mask`` then
    ``scatter``); a test convenience for callers that bin raw landing positions
    directly, without a detection chain in between.
    """
    pix_id, valid = sensor.pixel_index_and_mask(sensor_idx, x, y)
    return sensor.scatter(pix_id, valid, values)


# --- finite-difference slope --------------------------------------------------


def fd_slope(sag_fn, x, y, h=1e-5):
    """Central-difference (dz/dx, dz/dy) of a scalar sag function."""
    dzdx = (sag_fn(x + h, y) - sag_fn(x - h, y)) / (2 * h)
    dzdy = (sag_fn(x, y + h) - sag_fn(x, y - h)) / (2 * h)
    return float(dzdx), float(dzdy)
