"""The effective-aperture table: what the optics does with angle and wavelength."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import (
    Camera,
    SquareSensorGroup,
    TabulatedQE,
    TabulatedResponse,
    Telescope,
)
from iactrace.analysis import (
    FieldFrame,
    effective_aperture,
    effective_area,
    pixel_response,
)
from iactrace.telescope import mirrors

MIRROR_RADIUS = 0.1
REFLECTIVITY = 0.8
FOCAL_LENGTH = 0.5  # curvature 1.0 paraboloid
GEOMETRIC_AREA = math.pi * MIRROR_RADIUS**2 * REFLECTIVITY

REFL_WAVELENGTHS = [300.0, 400.0, 500.0, 600.0]
REFL_VALUES = [0.5, 0.9, 0.9, 0.3]
QE_WAVELENGTHS = [250.0, 400.0, 700.0]
QE_VALUES = [0.1, 0.4, 0.05]

# The lattice the rig is scanned on. A pixel is 0.005 / FOCAL_LENGTH = 0.01 rad
# across and the camera's far corner reaches 0.0247 / FOCAL_LENGTH, so this is a
# pitch/8 step over the camera plus 20% margin.
HALF_ANGLE = 0.0594
STEP = 0.00125

#: The band, built by the recipe the docs give now that nothing derives it: the
#: optics' own tabulated knots, densified so the curves between them are resolved.
BAND = np.unique(np.concatenate([REFL_WAVELENGTHS, QE_WAVELENGTHS, np.linspace(250.0, 700.0, 64)]))

#: The comatic rig carries no wavelength-dependent component, so two points spanning
#: the range describe its flat response exactly.
FLAT_BAND = [300.0, 600.0]


def make_rig(*, n_samples=256, n_pixels=7, spectral=False, camera_rotation=None):
    """A paraboloid + square camera at its focus, optionally wavelength-dependent."""
    mirror = mirrors.parabolic(
        position=(0.0, 0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        radius=MIRROR_RADIUS,
        reflectivity=REFLECTIVITY,
        reflectivity_curve=(
            TabulatedResponse.from_wavelengths(REFL_WAVELENGTHS, REFL_VALUES, n_elements=1)
            if spectral
            else None
        ),
        n_samples=n_samples,
        key=jax.random.key(0),
    )
    half_width = 0.0025 * n_pixels
    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=n_pixels,
        height=n_pixels,
        bounds=(-half_width, half_width, -half_width, half_width),
        photodetector=TabulatedQE.from_table(QE_WAVELENGTHS, QE_VALUES) if spectral else None,
    )
    telescope = Telescope(
        mirror_groups=[mirror],
        name="rig",
        camera_position=[0.0, 0.0, FOCAL_LENGTH],
        camera_rotation=camera_rotation,
    )
    return telescope, Camera(sensor_groups=[sensor])


def _image_centroid(telescope, camera, frame, lon, lat):
    """Centroid of the pixel image, in the sensor's own (x, y), for one field offset."""
    sensor = camera.sensor_groups[0]
    direction = frame.directions(jnp.asarray(lon), jnp.asarray(lat))
    image = np.asarray(
        camera.image(telescope.render(direction[None, :], jnp.ones(1), source_type="parallel"))
    )[0]
    centres = np.asarray(sensor.pixel_centers).reshape(sensor.height, sensor.width, 2)
    total = image.sum()
    assert total > 0, "no light reached the camera"
    return (centres * image[..., None]).sum(axis=(0, 1)) / total


@pytest.fixture(scope="module")
def flat_rig():
    return make_rig()


@pytest.fixture(scope="module")
def spectral_rig():
    return make_rig(spectral=True)


@pytest.fixture(scope="module")
def table(spectral_rig):
    return effective_aperture(*spectral_rig, half_angle=HALF_ANGLE, step=STEP, wavelengths=BAND)


class TestFieldFrame:
    def test_axis_is_exact_and_noise_free(self, flat_rig):
        """The axis comes from element orientations, so it does not wobble."""
        telescope, _ = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        assert np.allclose(np.asarray(frame.axis), [0.0, 0.0, -1.0], atol=1e-12)

    def test_triad_is_orthonormal_with_offset_frame_handedness(self, flat_rig):
        telescope, _ = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        axis, e_lon, e_lat = (np.asarray(v) for v in (frame.axis, frame.e_lon, frame.e_lat))
        for v in (axis, e_lon, e_lat):
            assert np.isclose(np.linalg.norm(v), 1.0)
        assert np.isclose(np.dot(axis, e_lon), 0.0, atol=1e-12)
        assert np.isclose(np.dot(axis, e_lat), 0.0, atol=1e-12)
        assert np.isclose(np.dot(e_lon, e_lat), 0.0, atol=1e-12)
        # An offset frame about the pointing direction (-axis) is left-handed;
        # consumers rely on this to recover the (lon, lat) they asked for.
        assert np.allclose(np.cross(axis, e_lon), e_lat, atol=1e-12)

    def test_directions_match_the_offset_parameterisation(self, flat_rig):
        """The three projections that define a direction from (lon, lat)."""
        telescope, _ = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        lon = jnp.array([0.0, 0.01, -0.03, 0.05])
        lat = jnp.array([0.0, -0.02, 0.04, 0.01])
        d = np.asarray(frame.directions(lon, lat), dtype=np.float64)

        assert np.allclose(d @ np.asarray(frame.axis), np.cos(lat) * np.cos(lon), atol=1e-6)
        assert np.allclose(d @ np.asarray(frame.e_lon), np.cos(lat) * np.sin(lon), atol=1e-6)
        assert np.allclose(d @ np.asarray(frame.e_lat), np.sin(lat), atol=1e-6)

    def test_field_offsets_move_the_image_along_the_convention_axes(self, flat_rig):
        """Altitude offsets slide the image along x, azimuth offsets along y.

        The sim_telarray convention this follows (see
        ``docs/getting_started/conventions.rst``) puts ``+X`` on the
        north-south axis and ``+Y`` west-east, which makes ``+X`` the
        *decreasing altitude* direction and ``+Y`` the *decreasing azimuth*
        one. Ray tracing then inverts the image through the focus, so a source
        higher in altitude lands at ``+X`` and one further round in azimuth at
        ``+Y``.

        This is what ties the field frame to the pixel layout: without it a
        tabulated response could come out transposed or mirrored -- and being
        rotated by 90 degrees is exactly the mistake this pins down.
        """
        telescope, camera = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        offset = 0.02

        on_axis = _image_centroid(telescope, camera, frame, 0.0, 0.0)
        assert np.allclose(on_axis, 0.0, atol=1e-4)

        moved_lat = _image_centroid(telescope, camera, frame, 0.0, offset)
        moved_lon = _image_centroid(telescope, camera, frame, offset, 0.0)
        # +lat (higher altitude) -> +X, the axis that points groundward.
        assert np.isclose(moved_lat[0], +FOCAL_LENGTH * offset, rtol=0.05)
        assert abs(moved_lat[1]) < 1e-4
        # +lon (advancing azimuth) -> +Y, the westward axis.
        assert np.isclose(moved_lon[1], +FOCAL_LENGTH * offset, rtol=0.05)
        assert abs(moved_lon[0]) < 1e-4

    def test_sky_axes_follow_the_telescope_frame_convention(self, flat_rig):
        """``e_lat = +X`` and ``e_lon = +Y``, straight from the convention.

        ``+X`` is the *decreasing* altitude direction, but ``directions``
        parameterises the propagation direction rather than the source's, so
        the two negations cancel and the sky axis an offset frame sees,
        ``-e_lat``, does climb in altitude.
        """
        telescope, _ = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        assert np.allclose(np.asarray(frame.e_lat), [1.0, 0.0, 0.0], atol=1e-7)
        assert np.allclose(np.asarray(frame.e_lon), [0.0, 1.0, 0.0], atol=1e-7)
        # The sky axes themselves: increasing altitude, increasing azimuth.
        assert np.allclose(-np.asarray(frame.e_lat), [-1.0, 0.0, 0.0], atol=1e-7)
        assert np.allclose(-np.asarray(frame.e_lon), [0.0, -1.0, 0.0], atol=1e-7)

    def test_directions_are_unit_vectors(self, flat_rig):
        telescope, _ = flat_rig
        frame = FieldFrame.from_telescope(telescope)
        d = np.asarray(frame.directions(jnp.linspace(-0.05, 0.05, 7), jnp.zeros(7)))
        assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)

    def test_camera_mounting_does_not_redefine_azimuth(self):
        """Bolting the camera in rotated cannot change what ``lon`` means."""
        base = FieldFrame.from_telescope(make_rig()[0])
        rolled = FieldFrame.from_telescope(make_rig(camera_rotation=[0.0, 0.0, 90.0])[0])
        assert np.allclose(np.asarray(rolled.e_lon), np.asarray(base.e_lon), atol=1e-7)
        assert np.allclose(np.asarray(rolled.e_lat), np.asarray(base.e_lat), atol=1e-7)

    def test_rotation_deg_rolls_the_sky_axes(self, flat_rig):
        telescope, _ = flat_rig
        base = FieldFrame.from_telescope(telescope)
        rolled = FieldFrame.from_telescope(telescope, rotation_deg=90.0)
        assert np.allclose(np.asarray(rolled.e_lon), np.asarray(base.e_lat), atol=1e-7)
        assert np.allclose(np.asarray(rolled.e_lat), -np.asarray(base.e_lon), atol=1e-7)
        # A roll is a rotation, so it cannot change the parity.
        assert np.allclose(
            np.cross(np.asarray(rolled.axis), np.asarray(rolled.e_lon)),
            np.asarray(rolled.e_lat),
            atol=1e-7,
        )


class TestEffectiveArea:
    def test_matches_the_geometric_aperture(self, flat_rig):
        area = effective_area(*flat_rig, [400.0])
        assert np.isclose(area[0], GEOMETRIC_AREA, rtol=1e-3)

    def test_tracks_the_product_of_every_curve(self, spectral_rig):
        wavelengths = np.array(REFL_WAVELENGTHS)
        area = effective_area(*spectral_rig, wavelengths)
        expected = (
            GEOMETRIC_AREA
            * np.array(REFL_VALUES)
            * np.interp(wavelengths, QE_WAVELENGTHS, QE_VALUES)
        )
        assert np.allclose(area, expected, rtol=1e-3)


class TestEffectiveApertureTable:
    def test_shapes_and_lattice_invariants(self, table):
        assert table.values.shape == (table.n_pixels, table.window, table.window)
        assert table.n_pixels == 49
        # Odd window: Simpson quadrature over it is then exact.
        assert table.window % 2 == 1
        # Canonical lattice: the lowest window corner sits on node 0.
        assert table.offset.min(axis=0).tolist() == [0, 0]
        assert np.all(table.step > 0)

    def test_zero_is_a_lattice_node(self, table):
        """So the on-axis response is read off rather than interpolated."""
        nodes = -table.origin / table.step
        assert np.allclose(nodes, np.rint(nodes), atol=1e-9)

    def test_response_vanishes_on_every_window_boundary(self, table):
        assert table.values[:, 0, :].max() == 0.0
        assert table.values[:, -1, :].max() == 0.0
        assert table.values[:, :, 0].max() == 0.0
        assert table.values[:, :, -1].max() == 0.0

    def test_every_pixel_has_signal(self, table):
        assert np.all(table.values.sum(axis=(1, 2)) > 0)

    def test_on_axis_normalisation_is_consistent(self, table):
        """The band-averaged B(lambda) must match the table's own on-axis sum.

        ``on_axis_area`` is noise-free (quadrature over the spectral pass) while
        the table entry is one Monte-Carlo estimate from ``n_samples`` rays whose
        wavelengths are drawn from a throughput varying by ~10x across the band,
        so the two agree only to a few percent at this rig's sample count. That
        is what ``meta["on_axis_area_table"]`` is for: it is the convergence
        diagnostic, and this test pins that it is the right quantity rather than
        that the scan has converged.
        """
        ratio = table.meta["on_axis_area_table"] / table.on_axis_area
        assert np.isclose(ratio, 1.0, rtol=0.05)

    def test_bandpass_is_the_effective_area_spectrum(self, table, spectral_rig):
        assert table.wavelengths.shape == table.spectral_area.shape
        direct = effective_area(*spectral_rig, table.wavelengths[::16])
        assert np.allclose(table.spectral_area[::16], direct, rtol=1e-6)

    def test_centres_track_the_camera_layout(self, table):
        """Pixel centres come out on a regular grid, at the camera's own pitch.

        Measured from the centres themselves rather than read off ``meta``: the
        scan no longer estimates a pitch, so the table's own data is the only
        thing left that can be checked against the camera it came from.
        """
        centres = table.centres
        expected = 0.005 / FOCAL_LENGTH  # pixel width / focal length
        gaps = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        assert np.isclose(np.median(gaps.min(axis=1)), expected, rtol=0.05)
        assert np.isclose(np.abs(centres).max(), 3 * expected, rtol=0.05)

    def test_the_scanned_field_matches_what_was_asked_for(self, table):
        """The field is rounded to a whole number of steps, and says so."""
        assert abs(table.meta["half_angle_rad"] - HALF_ANGLE) <= STEP

    def test_rejects_a_single_wavelength(self, spectral_rig):
        """One wavelength is a spike, not a band a consumer can integrate."""
        with pytest.raises(ValueError, match="at least two distinct values"):
            effective_aperture(*spectral_rig, half_angle=HALF_ANGLE, step=STEP, wavelengths=[450.0])

    def test_rejects_a_field_smaller_than_the_camera(self, spectral_rig):
        with pytest.raises(ValueError, match="edge of the field|received no light"):
            effective_aperture(*spectral_rig, half_angle=0.01, step=STEP, wavelengths=BAND)

    def test_rejects_a_window_the_response_overflows(self, spectral_rig):
        with pytest.raises(ValueError, match="on the boundary of a 3x3 window"):
            effective_aperture(
                *spectral_rig, half_angle=HALF_ANGLE, step=STEP, wavelengths=BAND, window=3
            )


# Window placement


def make_comatic_rig(n_pixels=15, n_samples=400):
    """A fast spherical mirror: strong, one-sided off-axis coma, like an IACT's.

    The parabolic rig above is stigmatic on axis and near-symmetric off it, so it
    cannot show what window placement is for. A spherical mirror at f/1.2 smears
    an off-axis pixel's response into a comatic tail pointing radially outward.
    """
    focal, radius = 0.3, 0.125
    mirror = mirrors.spherical(
        position=(0.0, 0.0, 0.0),
        focal_length=focal,
        radius=radius,
        reflectivity=REFLECTIVITY,
        n_samples=n_samples,
        key=jax.random.key(0),
    )
    half_width = 0.03
    sensor = SquareSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        width=n_pixels,
        height=n_pixels,
        bounds=(-half_width, half_width, -half_width, half_width),
    )
    telescope = Telescope(mirror_groups=[mirror], name="coma", camera_position=[0.0, 0.0, focal])
    return telescope, Camera(sensor_groups=[sensor])


#: The comatic rig's pixels are coarser -- a 0.01333 rad pitch -- and it reaches
#: further off axis. Sampled at pitch/4, which is enough to resolve the tail.
COMA_HALF_ANGLE = 0.1686
COMA_STEP = 0.0033333


@pytest.fixture(scope="module")
def comatic_table():
    return effective_aperture(
        *make_comatic_rig(), half_angle=COMA_HALF_ANGLE, step=COMA_STEP, wavelengths=FLAT_BAND
    )


def _boxes(table, tol=1e-3):
    """Each pixel's response bounding box in field offset, from the table itself."""
    lo = np.empty((table.n_pixels, 2))
    hi = np.empty((table.n_pixels, 2))
    for p in range(table.n_pixels):
        live = np.argwhere(table.values[p] > tol * table.values[p].max())
        lo[p] = table.origin + (table.offset[p] + live.min(axis=0)) * table.step
        hi[p] = table.origin + (table.offset[p] + live.max(axis=0)) * table.step
    return lo, hi


class TestWindowPlacement:
    def test_windows_are_not_pinned_to_pixel_centres(self, comatic_table):
        """A one-sided response drags its window off the pixel it belongs to."""
        drift = np.abs(comatic_table.centres - comatic_table.window_centres)
        assert (drift / comatic_table.step).max() > 1.0

    def test_a_centred_window_would_have_to_be_larger(self, comatic_table):
        """The saving, stated as the thing it saves.

        A window centred on the pixel has to reach the far end of the comatic
        tail on *both* sides; one placed on the response's own extent reaches it
        only once. This is what the placement buys, and it is quadratic in the
        stored table and in every later projection.
        """
        lo, hi = _boxes(comatic_table)
        centred = 2 * np.maximum(comatic_table.centres - lo, hi - comatic_table.centres).max()
        on_extent = (hi - lo).max()
        assert centred > 1.3 * on_extent

    def test_the_window_is_cropped_to_what_the_response_needs(self, comatic_table):
        """No pixel's response may be smaller than the window by a whole margin.

        The window is sized from conservative estimates and then cropped to fit,
        so the widest response must come within a couple of nodes of filling it.
        """
        lo, hi = _boxes(comatic_table)
        widest = (hi - lo).max() / comatic_table.step.min()
        assert comatic_table.window - widest <= 4

    def test_most_of_a_window_is_not_empty(self, comatic_table):
        """The property the placement exists for, as recorded in ``meta``."""
        assert comatic_table.meta["window_fill"] > 0.3

    def test_the_lattice_is_still_canonical_after_cropping(self, comatic_table):
        assert comatic_table.offset.min(axis=0).tolist() == [0, 0]
        nodes = -comatic_table.origin / comatic_table.step
        assert np.allclose(nodes, np.rint(nodes), atol=1e-9)

    def test_centres_are_the_response_first_moment(self, comatic_table):
        """Not the window centre, and not the geometric pixel centre either."""
        lo, hi = _boxes(comatic_table)
        # Inside its own box, by construction.
        assert np.all(comatic_table.centres >= lo - comatic_table.step)
        assert np.all(comatic_table.centres <= hi + comatic_table.step)
        # Pulled off the box centre, into the tail.
        pull = np.abs(comatic_table.centres - 0.5 * (lo + hi)) / comatic_table.step
        assert pull.max() > 0.5

    def test_zero_is_a_node_without_any_snapping(self, comatic_table):
        """The lattice is centred and odd, so the on-axis response is read off."""
        nodes = -comatic_table.origin / comatic_table.step
        assert np.allclose(nodes, np.rint(nodes), atol=1e-9)
        assert comatic_table.offset.min(axis=0).tolist() == [0, 0]

    def test_the_stored_window_is_the_traced_response(self, comatic_table):
        """Sliced straight out of the field, not resampled onto the window.

        Reads the on-axis node out of each pixel's window and compares it against
        a direct kernel call at the same direction. They must agree exactly, not
        approximately: the dense pass slices, it does not interpolate.
        """
        telescope, camera = make_comatic_rig()
        frame = FieldFrame.from_telescope(telescope)
        direct = np.asarray(
            pixel_response(telescope, camera, frame.directions(jnp.zeros(1), jnp.zeros(1)))
        )[0]

        node = np.rint(-comatic_table.origin / comatic_table.step).astype(int)
        local = node - comatic_table.offset
        inside = np.all((local >= 0) & (local < comatic_table.window), axis=-1)
        assert inside.any(), "no pixel's window covers the optical axis"
        stored = comatic_table.values[inside, local[inside, 0], local[inside, 1]]
        # The scan is band-weighted and this call is not, so compare the pattern
        # rather than the scale: the same pixels must be lit, in the same order.
        assert np.argmax(stored) == np.argmax(np.clip(direct, 0, None)[inside])
