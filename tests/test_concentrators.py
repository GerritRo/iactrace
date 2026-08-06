import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Camera, ConstantQE, HexagonalSensorGroup, OkumuraCone, WinstonCone
from iactrace.camera.optics.okumura import _bezier_power_coeffs, _polyval
from iactrace.camera.optics.winston import cpc_full_length, cpc_wall_tilt, profile_apothem
from iactrace.core.ray_bundle import RayBundle

from ._helpers import make_hex_centers

# Optimized relative control points from Okumura (2012), Table 1
# (rho1 = 20 mm, rho2 = 10 mm, R = 1.0, n = 1.0).
QUAD_P1 = (0.89, 0.35)
CUBIC_P1, CUBIC_P2 = (0.39, 0.18), (0.87, 0.36)

CONE_KINDS = ["winston", "okumura"]


@pytest.fixture(params=CONE_KINDS)
def cone_kind(request):
    return request.param


def _cone_from_apertures(kind, entrance_apothem, exit_apothem, n_sides=6, **kwargs):
    """A hexagonal cone of the given kind from its two aperture apothems."""
    if kind == "winston":
        return WinstonCone(n_sides, entrance_apothem, exit_apothem, **kwargs)
    return OkumuraCone.quadratic(n_sides, entrance_apothem, exit_apothem, QUAD_P1, **kwargs)


def _make_cone(kind, exit_apothem, cutoff_deg, **kwargs):
    """A cone whose design cutoff sets the mouth apothem a1 = a2 / sin(theta_c)."""
    a1 = exit_apothem / math.sin(math.radians(cutoff_deg))
    return _cone_from_apertures(kind, a1, exit_apothem, **kwargs)


def _fill_entrance(cone, n, seed=0):
    """Random points uniformly inside the cone entrance polygon (z=0 chain)."""
    rng = np.random.default_rng(seed)
    a = cone.entrance_apothem
    n_hats = np.asarray(cone._wall_normals())
    pts = np.empty((n, 2))
    filled = 0
    while filled < n:
        xy = rng.uniform(-a, a, (4 * n, 2))
        ok = np.all(xy @ n_hats.T <= a + 1e-12, axis=1)
        good = xy[ok]
        take = min(len(good), n - filled)
        pts[filled : filled + take] = good[:take]
        filled += take
    return pts


def _launch(cone, alpha_deg, n=1000, seed=0):
    """Scatter ``n`` rays entering the mouth at polar incidence ``alpha_deg``."""
    rng = np.random.default_rng(seed + 1)
    xy = _fill_entrance(cone, n, seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    a = math.radians(alpha_deg)
    origins = jnp.asarray(np.concatenate([xy, np.zeros((n, 1))], axis=1))
    # light travels -z into the chain (entrance at z=0)
    dirs = jnp.asarray(
        np.stack(
            [np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.full(n, -np.cos(a))],
            axis=1,
        )
    )
    rb = RayBundle(
        origins=origins,
        directions=dirs,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )
    return cone.apply(rb)


def _transmission(cone, alpha_deg, n=1000):
    return float((np.asarray(_launch(cone, alpha_deg, n).values) > 0).mean())


# 1. Cone-specific geometry
# -----------------------------------------------------------------------------


class TestWinstonGeometry:
    """CPC parabola relations -- specific to the Winston cone."""

    def test_aperture_and_length_relations(self):
        a2 = 0.5
        for cutoff_deg in (15.0, 20.0, 30.0):
            s_d = math.sin(math.radians(cutoff_deg))
            c_d = math.cos(math.radians(cutoff_deg))
            a1 = a2 / s_d  # full mouth
            length = cpc_full_length(a2, s_d, c_d)
            assert length == pytest.approx((a1 + a2) * c_d / s_d)
            # the wall tilt recovered from the three physical dims matches design
            s, c = cpc_wall_tilt(a2, a1, length)
            assert s == pytest.approx(s_d)
            assert c == pytest.approx(c_d)

    def test_profile_endpoints_and_monotonic(self):
        a2 = 0.5
        s, c = math.sin(math.radians(20.0)), math.cos(math.radians(20.0))
        a1 = a2 / s
        length = cpc_full_length(a2, s, c)
        z = jnp.linspace(0.0, length, 50)
        r = profile_apothem(z, a2, s, c)
        assert float(r[0]) == pytest.approx(a2, abs=1e-6)  # exit
        assert float(r[-1]) == pytest.approx(a1, abs=1e-6)  # entrance
        assert bool(jnp.all(jnp.diff(r) >= -1e-6))  # monotonic

    def test_constructors_and_truncation(self):
        a2 = 0.5
        s_d, c_d = math.sin(math.radians(20.0)), math.cos(math.radians(20.0))
        a1 = a2 / s_d  # full mouth
        length = cpc_full_length(a2, s_d, c_d)

        full = WinstonCone(6, a1, a2)  # entrance, exit (full)
        assert full.length == pytest.approx(length)
        assert full.entrance_apothem == pytest.approx(a1, abs=1e-6)
        assert full.s == pytest.approx(s_d, abs=1e-6)

        # Truncated cone given by its PHYSICAL mouth at z = length: the wall tilt
        # (hence the parabola) is recovered exactly from (a2, a1, length).
        phys = float(profile_apothem(jnp.asarray(2.0), a2, s_d, c_d))
        assert phys < a1  # truncated mouth is smaller
        trunc = WinstonCone(6, phys, a2, length=2.0)
        assert trunc.length == pytest.approx(2.0)
        assert trunc.entrance_apothem == pytest.approx(phys)  # mouth == what we passed
        assert trunc.s == pytest.approx(s_d, abs=1e-6)  # same parabola

        with pytest.raises(ValueError):
            WinstonCone(6, 0.5, 0.6)  # exit > entrance
        with pytest.raises(ValueError):
            # mouth far too wide for this shallow depth -> not a realizable CPC
            WinstonCone(6, a1, a2, length=0.05)


class TestOkumuraGeometry:
    """Bezier-meridian relations -- specific to the Okumura cone."""

    def test_degree_and_control_points(self):
        q = OkumuraCone.quadratic(6, 0.02, 0.01, QUAD_P1)
        c = OkumuraCone.cubic(6, 0.02, 0.01, CUBIC_P1, CUBIC_P2)
        assert q.degree == 2 and c.degree == 3
        assert q.control_points == (QUAD_P1,)
        assert c.control_points == (CUBIC_P1, CUBIC_P2)

    def test_default_length_matches_paper_and_winston(self):
        # length=None must reproduce the Winston-equivalent depth exactly, so the
        # Okumura cone is a genuine drop-in for the cone it compares against; and
        # the rho1=20mm, rho2=10mm cone reproduces Okumura Table 2's L = 52.0 mm.
        for a2, cutoff in [(0.010, 30.0), (0.010, 20.0), (0.005, 25.0)]:
            a1 = a2 / math.sin(math.radians(cutoff))
            win = WinstonCone(6, a1, a2)
            bez = OkumuraCone.quadratic(6, a1, a2, QUAD_P1)
            assert bez.length == pytest.approx(win.length, rel=1e-9)
        assert OkumuraCone.quadratic(6, 0.020, 0.010, QUAD_P1).length == pytest.approx(
            0.0520, abs=5e-5
        )

    def test_meridian_and_cross_section_endpoints(self):
        a1, a2 = 0.020, 0.010
        bez = OkumuraCone.cubic(6, a1, a2, CUBIC_P1, CUBIC_P2)
        r_c = jnp.asarray(bez.r_coeffs)
        z_c = jnp.asarray(bez.z_coeffs)
        # t=0 -> exit rim (a2, z=0); t=1 -> mouth (a1, z=length)
        assert float(_polyval(r_c, jnp.asarray(0.0))) == pytest.approx(a2)
        assert float(_polyval(r_c, jnp.asarray(1.0))) == pytest.approx(a1)
        assert float(_polyval(z_c, jnp.asarray(0.0))) == pytest.approx(0.0, abs=1e-12)
        assert float(_polyval(z_c, jnp.asarray(1.0))) == pytest.approx(bez.length)
        # cross_sections() feeds the mesh path: entrance inradius a1, exit a2.
        z, rings = bez.cross_sections()
        assert rings.shape[1:] == (6, 2)
        inrad_entrance = float(jnp.linalg.norm(rings[0, 0])) * math.cos(math.pi / 6)
        inrad_exit = float(jnp.linalg.norm(rings[-1, 0])) * math.cos(math.pi / 6)
        assert inrad_entrance == pytest.approx(a1, rel=1e-6)
        assert inrad_exit == pytest.approx(a2, rel=1e-6)

    def test_invalid_constructions(self):
        with pytest.raises(ValueError):
            OkumuraCone(6, 0.01, 0.02, [QUAD_P1])  # exit > entrance
        with pytest.raises(ValueError):
            OkumuraCone(6, 0.02, 0.01, [])  # no control point
        with pytest.raises(ValueError):
            OkumuraCone(6, 0.02, 0.01, [(0.5, -0.3)])  # non-monotonic Z(t)

    def test_bezier_power_coeffs(self):
        # Straight control values -> identity B(t) = t; a quadratic through
        # (0, 1/2, 1) is also the line t (midpoint control gives no curvature).
        assert _bezier_power_coeffs([0.0, 1.0]) == pytest.approx([0.0, 1.0])
        assert _bezier_power_coeffs([0.0, 0.5, 1.0]) == pytest.approx([0.0, 1.0, 0.0])

    @pytest.mark.slow
    def test_beats_winston(self):
        # Okumura's headline result: same (a1, a2, L), but the Okumura cone's
        # Bezier walls collect more signal (theta < theta_max) and reject more
        # stray light (theta_max < theta < 1.5 theta_max) than Winston's paraboloid.
        a1, a2, tmax = 0.020, 0.010, 30.0
        win = WinstonCone(6, a1, a2, reflectivity=1.0, max_bounces=40)
        cub = OkumuraCone.cubic(6, a1, a2, CUBIC_P1, CUBIC_P2, reflectivity=1.0, max_bounces=40)

        def band(cone, angles):
            return float(
                np.mean([np.asarray(_launch(cone, a, n=1500).values).sum() / 1500 for a in angles])
            )

        signal = np.arange(0.0, tmax, 5.0)
        background = np.arange(tmax, 1.5 * tmax + 0.1, 5.0)
        assert band(cub, signal) > band(win, signal)  # more signal
        assert band(cub, background) < band(win, background)  # less background


# 2. Shared physics: acceptance, energy, mouth aperture
# -----------------------------------------------------------------------------


class TestAcceptance:
    """Angular acceptance -- the defining light-collector property (both walls)."""

    @pytest.mark.slow
    def test_cutoff_near_acceptance_angle(self, cone_kind):
        cone = _make_cone(cone_kind, 0.5, 20.0, reflectivity=1.0, max_bounces=40)
        # well inside -> almost everything transmits
        assert _transmission(cone, 0.0) > 0.95
        assert _transmission(cone, 12.0) > 0.9
        # around the design angle -> roughly half
        assert 0.35 < _transmission(cone, 20.0) < 0.8
        # well outside -> strongly rejected
        assert _transmission(cone, 28.0) < 0.25
        assert _transmission(cone, 40.0) < 0.05


class TestEnergy:
    """Throughput never amplifies and stays bounded by reflectivity^bounces.

    Wall-shape-independent, so exercised on the Winston cone alone.
    """

    def test_no_gain_bounded_and_unit_preserving(self):
        cone = _make_cone("winston", 0.5, 20.0, reflectivity=0.9, max_bounces=40)
        v = np.asarray(_launch(cone, 8.0, n=1500).values)
        assert v.max() <= 1.0 + 1e-9  # never amplifies
        transmitted = v[v > 0]
        assert transmitted.min() >= 0.9**40 - 1e-12  # >= reflectivity^max_bounces
        assert transmitted.mean() < 1.0  # some bounces happened

        # With perfect reflectivity every transmitted ray keeps unit value.
        perfect = _make_cone("winston", 0.5, 20.0, reflectivity=1.0, max_bounces=40)
        vp = np.asarray(_launch(perfect, 5.0, n=1000).values)
        assert set(np.unique(np.round(vp, 9)).tolist()) <= {0.0, 1.0}

    def test_exit_plane_and_path_length(self):
        cone = _make_cone("winston", 0.5, 20.0, reflectivity=1.0, max_bounces=40)
        out = _launch(cone, 6.0, n=1000)
        v = np.asarray(out.values)
        # Transmitted rays sit on the exit plane; dead rays' origins are meaningless.
        assert bool(np.allclose(np.asarray(out.origins)[v > 0, 2], -cone.length))
        assert bool(np.all(np.asarray(out.alive) == (v > 0)))
        # transmitted rays travel at least the cone depth
        assert np.asarray(out.path_length)[v > 0].min() >= cone.length - 1e-9


class TestMouthAperture:
    def test_rays_outside_mouth_are_lost(self):
        # apply() masks rays entering outside the entrance polygon: a ray just
        # inside the mouth transmits, one far outside is zeroed. Masking is
        # wall-shape-independent, so one representative cone suffices.
        cone = _make_cone("winston", 0.5, 20.0, reflectivity=1.0)
        a = cone.entrance_apothem
        origins = jnp.array([[0.0, 0.0, 0.0], [10 * a, 0.0, 0.0]])
        dirs = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (2, 1))  # light -z
        rb = RayBundle(
            origins=origins,
            directions=dirs,
            values=jnp.ones(2),
            path_length=jnp.zeros(2),
            n=jnp.ones(2),
        )
        out = cone.apply(rb)
        assert float(out.values[0]) > 0.0
        assert float(out.values[1]) == 0.0


# 3. Chain integration + rotation alignment
# -----------------------------------------------------------------------------


def _hex_camera(kind, with_cone):
    centers = make_hex_centers(n_rings=2, hex_size=0.02)
    entrance = float(
        HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=centers,
        ).hex_inradius
    )
    cone = (
        _cone_from_apertures(kind, entrance, 0.35 * entrance, reflectivity=0.95, max_bounces=20)
        if with_cone
        else None
    )
    sensor = HexagonalSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        hex_centers=centers,
        concentrator=cone,
        photodetector=ConstantQE(0.9),
    )
    return Camera([sensor]), sensor


def _downward(xy, z=0.05):
    xy = jnp.asarray(xy, dtype=float)
    n = xy.shape[0]
    origins = jnp.concatenate([xy, jnp.full((n, 1), z)], axis=1)
    dirs = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
    return RayBundle(
        origins=origins,
        directions=dirs,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


class TestChainIntegration:
    def test_image_and_collect_run(self):
        # Camera/cone plumbing is wall-shape-independent -> one representative cone.
        cam, sensor = _hex_camera("winston", with_cone=True)
        rb = _downward(np.zeros((50, 2)))  # on-axis rays into centre pixel
        img = cam.image(rb)
        assert img.shape == (1, sensor.n_pixels)
        assert float(img.sum()) > 0.0
        pe, t, pix, hit = cam.collect(rb)
        assert pe.shape == (50,) and t.shape == (50,)
        assert bool((np.asarray(pe) <= 0.9 + 1e-9).all())  # <= QE, attenuated
        # a cone can only attenuate relative to no cone
        without, _ = _hex_camera("winston", with_cone=False)
        assert float(img.sum()) <= float(without.image(rb).sum()) + 1e-9


def _rot2(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(pts) @ np.array([[c, -s], [s, c]]).T


class TestRotationAlignment:
    """The cone must stay aligned with its pixel under any hex grid rotation.

    ``to_pixel_frame`` normalises ``grid_rotation`` away, so the cone (defined in
    the grid-aligned pixel frame) co-rotates with the layout. Alignment is a
    frame-transform property, exercised on one representative cone.
    """

    def _setup(self, kind):
        size = 0.02
        centers0 = np.asarray(make_hex_centers(2, size))
        inrad = float(HexagonalSensorGroup([[0, 0, 0]], [[0, 0, 0]], centers0).hex_inradius)
        cone = _cone_from_apertures(
            kind, inrad, np.sin(np.radians(20.0)) * inrad, reflectivity=1.0, max_bounces=60
        )
        rng = np.random.default_rng(1)
        n = 2000
        r = 0.8 * inrad * np.sqrt(rng.uniform(0, 1, n))  # fill only the central pixel
        psi = rng.uniform(0, 2 * np.pi, n)
        xy0 = np.c_[r * np.cos(psi), r * np.sin(psi)]
        al = np.radians(21.0)  # just past the 20 deg cutoff
        d0 = np.tile([np.sin(al), 0.0, -np.cos(al)], (n, 1))
        o0 = np.c_[xy0, np.full(n, 0.05)]
        return centers0, cone, o0, d0, n

    def _transmission(self, centers0, cone, o0, d0, n, theta):
        sensor = HexagonalSensorGroup(
            [[0, 0, 0]],
            [[0, 0, 0]],
            _rot2(centers0, theta),
            concentrator=cone,
            photodetector=ConstantQE(1.0),
        )
        cam = Camera([sensor])
        rb = RayBundle(
            origins=jnp.asarray(np.c_[_rot2(o0[:, :2], theta), o0[:, 2]]),
            directions=jnp.asarray(np.c_[_rot2(d0[:, :2], theta), d0[:, 2]]),
            values=jnp.ones(n),
            path_length=jnp.zeros(n),
            n=jnp.ones(n),
        )
        pe, _t, _pix, _hit = cam.collect(rb)
        return float(np.asarray(pe).sum())

    @pytest.mark.slow
    def test_transmission_invariant_under_layout_rotation(self):
        centers0, cone, o0, d0, n = self._setup("winston")
        base = self._transmission(centers0, cone, o0, d0, n, 0.0)
        assert 0.1 * n < base < 0.9 * n  # genuinely partial -> discriminating
        for deg in (11.0, 37.0):
            rotated = self._transmission(centers0, cone, o0, d0, n, np.radians(deg))
            assert abs(rotated - base) < 1e-6 * n

    def test_orientation_offset_changes_transmission(self):
        # Control: the test is sensitive -- offsetting the cone orientation
        # relative to the pixel changes the result (so invariance above is real).
        centers0, cone, o0, d0, n = self._setup("winston")
        offset = _cone_from_apertures(
            "winston",
            cone.entrance_apothem,
            cone.exit_apothem,
            reflectivity=1.0,
            max_bounces=60,
            orientation_deg=30.0,
        )
        base = self._transmission(centers0, cone, o0, d0, n, 0.0)
        shifted = self._transmission(centers0, offset, o0, d0, n, 0.0)
        assert abs(shifted - base) > 0.02 * n


# 4. Numerical robustness (single representative cone)
# -----------------------------------------------------------------------------


class TestRobustness:
    def test_apply_gradients_are_finite(self):
        # The trace's divisions/sqrts must be grad-safe (double-where), including
        # the degenerate horizontal ray (d[2] == 0) that hits the inf branch.
        cone = _cone_from_apertures("winston", 0.025, 0.01, reflectivity=0.9, max_bounces=8)

        def loss(o, d):
            m = o.shape[0]
            rb = RayBundle(o, d, jnp.ones(m), jnp.zeros(m), jnp.ones(m))
            out = cone.apply(rb)
            return jnp.sum(out.values) + jnp.sum(out.path_length)

        o = jnp.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
        d = jnp.array([[1.0, 0.0, 0.0], [0.1, 0.0, -0.99]])  # 1st ray is horizontal
        d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
        go, gd = jax.grad(loss, argnums=(0, 1))(o, d)
        assert bool(jnp.all(jnp.isfinite(go)))
        assert bool(jnp.all(jnp.isfinite(gd)))

    def test_small_cone_traces_in_range(self):
        # The accept/reject floor scales with a2, so a sub-mm cone still
        # transmits finite values in (0, 1] rather than absorbing everything.
        # (Kept separate from the gradient test: this guards small-scale numerics,
        # that one guards autodiff safety -- different failure modes.)
        cone = _cone_from_apertures("winston", 1.25e-4, 5e-5, reflectivity=0.95, max_bounces=12)
        rng = np.random.default_rng(0)
        xy = rng.uniform(-0.7, 0.7, (1000, 2)) * 1.25e-4
        rb = RayBundle(
            jnp.asarray(np.c_[xy, np.zeros(len(xy))]),
            jnp.tile(jnp.array([0.0, 0.0, -1.0]), (len(xy), 1)),
            jnp.ones(len(xy)),
            jnp.zeros(len(xy)),
            jnp.ones(len(xy)),
        )
        v = np.asarray(cone.apply(rb).values)
        assert np.all(np.isfinite(v))
        assert v.max() <= 1.0 + 1e-6
        assert v.mean() > 0.0


# 5. Rendering smoke (both mesh paths: Winston._meridian, Okumura.cross_sections)
# -----------------------------------------------------------------------------


class TestShowSensorChain:
    def test_show_sensor_chain_smoke(self, cone_kind):
        pytest.importorskip("trimesh")
        from iactrace import show_sensor_chain

        cam, _ = _hex_camera(cone_kind, with_cone=True)
        scene = show_sensor_chain(cam)
        assert len(scene.geometry) >= 3  # entrance + faceted cone + detector
