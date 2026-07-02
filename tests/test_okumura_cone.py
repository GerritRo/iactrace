import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Camera, HexagonalSensorGroup, OkumuraCone, UniformQE, WinstonCone
from iactrace.camera.okumura_cone import _bezier_power_coeffs, _polyval
from iactrace.core.ray_bundle import RayBundle

from .test_sensors import make_hex_centers

# Optimized relative control points from Okumura (2012), Table 1, for
# rho1 = 20 mm, rho2 = 10 mm, R = 1.0, n = 1.0.
QUAD_P1 = (0.89, 0.35)
CUBIC_P1, CUBIC_P2 = (0.39, 0.18), (0.87, 0.36)


def _quad(exit_apothem, cutoff_deg, n_sides=6, **kwargs):
    """Quadratic Okumura cone with the given exit apothem and design cutoff."""
    a1 = exit_apothem / math.sin(math.radians(cutoff_deg))
    return OkumuraCone.quadratic(n_sides, a1, exit_apothem, QUAD_P1, **kwargs)


def _cubic(exit_apothem, cutoff_deg, n_sides=6, **kwargs):
    a1 = exit_apothem / math.sin(math.radians(cutoff_deg))
    return OkumuraCone.cubic(n_sides, a1, exit_apothem, CUBIC_P1, CUBIC_P2, **kwargs)


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


def _launch(cone, alpha_deg, n=3000, seed=0):
    """Scatter ``n`` rays entering the mouth at polar incidence ``alpha_deg``."""
    rng = np.random.default_rng(seed + 1)
    xy = _fill_entrance(cone, n, seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    a = math.radians(alpha_deg)
    origins = jnp.asarray(np.concatenate([xy, np.zeros((n, 1))], axis=1))
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
    return jax.jit(cone.apply)(rb)


def _transmission(cone, alpha_deg, n=3000):
    return float((np.asarray(_launch(cone, alpha_deg, n).values) > 0).mean())


# 1. Geometry


class TestGeometry:
    def test_degree_and_control_points(self):
        q = OkumuraCone.quadratic(6, 0.02, 0.01, QUAD_P1)
        c = OkumuraCone.cubic(6, 0.02, 0.01, CUBIC_P1, CUBIC_P2)
        assert q.degree == 2 and c.degree == 3
        assert q.control_points == (QUAD_P1,)
        assert c.control_points == (CUBIC_P1, CUBIC_P2)

    def test_default_length_matches_equivalent_winston(self):
        # length=None must reproduce the Winston-equivalent depth exactly, so the
        # Okumura cone is a genuine drop-in for the cone it compares against.
        for a2, cutoff in [(0.010, 30.0), (0.010, 20.0), (0.005, 25.0)]:
            a1 = a2 / math.sin(math.radians(cutoff))
            win = WinstonCone(6, a1, a2)
            bez = OkumuraCone.quadratic(6, a1, a2, QUAD_P1)
            assert bez.length == pytest.approx(win.length, rel=1e-9)

    def test_paper_length_value(self):
        # Okumura Table 2: rho1 = 20 mm, rho2 = 10 mm -> L = 52.0 mm.
        bez = OkumuraCone.quadratic(6, 0.020, 0.010, QUAD_P1)
        assert bez.length == pytest.approx(0.0520, abs=5e-5)

    def test_meridian_endpoints(self):
        a1, a2 = 0.020, 0.010
        bez = OkumuraCone.cubic(6, a1, a2, CUBIC_P1, CUBIC_P2)
        r_c = jnp.asarray(bez.r_coeffs)
        z_c = jnp.asarray(bez.z_coeffs)
        # t=0 -> exit rim (a2, z=0); t=1 -> mouth (a1, z=length)
        assert float(_polyval(r_c, jnp.asarray(0.0))) == pytest.approx(a2)
        assert float(_polyval(r_c, jnp.asarray(1.0))) == pytest.approx(a1)
        assert float(_polyval(z_c, jnp.asarray(0.0))) == pytest.approx(0.0, abs=1e-12)
        assert float(_polyval(z_c, jnp.asarray(1.0))) == pytest.approx(bez.length)

    def test_cross_sections_endpoints(self):
        a1, a2 = 0.020, 0.010
        bez = OkumuraCone.quadratic(6, a1, a2, QUAD_P1)
        z, rings = bez.cross_sections()
        assert z.shape[0] == rings.shape[0]
        assert rings.shape[1:] == (6, 2)
        # entrance slice at z=0 with mouth inradius a1, exit slice at z=-length.
        assert float(z[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(z[-1]) == pytest.approx(-bez.length, abs=1e-9)
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


# 2. Angular acceptance (the defining light-collector property)


class TestAcceptance:
    def test_cutoff_near_acceptance_angle(self):
        cone = _quad(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        assert _transmission(cone, 0.0) > 0.95
        assert _transmission(cone, 12.0) > 0.9
        assert 0.35 < _transmission(cone, 20.0) < 0.8  # ~half at the design angle
        assert _transmission(cone, 28.0) < 0.2
        assert _transmission(cone, 40.0) < 0.03

    def test_monotone_decreasing(self):
        cone = _cubic(0.5, 25.0, reflectivity=1.0, max_bounces=40)
        ts = [_transmission(cone, a) for a in (0, 10, 20, 25, 30, 40)]
        assert all(ts[i] >= ts[i + 1] - 0.08 for i in range(len(ts) - 1))

    def test_beats_winston(self):
        # Okumura's headline result: same (a1, a2, L), but the Okumura cone's
        # Bezier walls collect more signal (theta < theta_max) and reject more
        # stray light (theta_max < theta < 1.5 theta_max) than Winston's paraboloid.
        a1, a2, tmax = 0.020, 0.010, 30.0
        win = WinstonCone(6, a1, a2, reflectivity=1.0, max_bounces=60)
        cub = OkumuraCone.cubic(6, a1, a2, CUBIC_P1, CUBIC_P2, reflectivity=1.0, max_bounces=60)

        def band(cone, angles):
            return float(
                np.mean([np.asarray(_launch(cone, a, n=2500).values).sum() / 2500 for a in angles])
            )

        signal = np.arange(0.0, tmax, 3.0)
        background = np.arange(tmax, 1.5 * tmax + 0.1, 3.0)
        assert band(cub, signal) > band(win, signal)  # more signal
        assert band(cub, background) < band(win, background)  # less background


# 3. Energy conservation


class TestEnergy:
    def test_no_gain_and_attenuated(self):
        cone = _quad(0.5, 20.0, reflectivity=0.9, max_bounces=40)
        out = _launch(cone, 8.0, n=4000)
        v = np.asarray(out.values)
        assert v.max() <= 1.0 + 1e-9  # never amplifies
        transmitted = v[v > 0]
        assert transmitted.min() >= 0.9**40 - 1e-12  # >= reflectivity^max_bounces
        assert transmitted.mean() < 1.0  # some bounces happened

    def test_reflectivity_one_keeps_unit_values(self):
        cone = _cubic(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        out = _launch(cone, 5.0, n=2000)
        v = np.asarray(out.values)
        assert set(np.unique(np.round(v, 9)).tolist()) <= {0.0, 1.0}

    def test_exit_plane_and_path_length(self):
        cone = _quad(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        out = _launch(cone, 6.0, n=1500)
        assert bool(np.allclose(np.asarray(out.origins)[:, 2], -cone.length))
        v = np.asarray(out.values)
        assert np.asarray(out.path_length)[v > 0].min() >= cone.length - 1e-9


# 4. Pure tracer + jit


class TestTracer:
    def test_rays_outside_mouth_are_lost(self):
        cone = _quad(0.5, 20.0, reflectivity=1.0)
        a = cone.entrance_apothem
        origins = jnp.array([[0.0, 0.0, 0.0], [10 * a, 0.0, 0.0]])
        dirs = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (2, 1))
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

    def test_jit(self):
        cone = _cubic(0.5, 20.0)
        rb = RayBundle(
            origins=jnp.zeros((5, 3)),
            directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (5, 1)),
            values=jnp.ones(5),
            path_length=jnp.zeros(5),
            n=jnp.ones(5),
        )
        out = jax.jit(cone.apply)(rb)
        assert out.values.shape == (5,)


# 5. Chain integration


def _hex_camera(with_cone):
    centers = make_hex_centers(n_rings=2, hex_size=0.02)
    entrance = float(
        HexagonalSensorGroup(
            positions=[[0.0, 0.0, 0.0]],
            rotations=[[0.0, 0.0, 0.0]],
            hex_centers=centers,
        ).hex_inradius
    )
    cone = (
        OkumuraCone.quadratic(
            6,
            entrance,
            0.35 * entrance,
            QUAD_P1,
            reflectivity=0.95,
            max_bounces=20,
        )
        if with_cone
        else None
    )
    sensor = HexagonalSensorGroup(
        positions=[[0.0, 0.0, 0.0]],
        rotations=[[0.0, 0.0, 0.0]],
        hex_centers=centers,
        concentrator=cone,
        photosensor=UniformQE(0.9),
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
        cam, sensor = _hex_camera(with_cone=True)
        rb = _downward(np.zeros((50, 2)))
        img = cam.image(rb)
        assert img.shape == (1, sensor.n_pixels)
        assert float(img.sum()) > 0.0
        pe, t, pix, hit = cam.collect(rb)
        assert pe.shape == (50,) and t.shape == (50,)
        assert bool((np.asarray(pe) <= 0.9 + 1e-9).all())

    def test_cone_attenuates_relative_to_no_cone(self):
        rb = _downward(np.zeros((200, 2)))
        with_cone, _ = _hex_camera(with_cone=True)
        without, _ = _hex_camera(with_cone=False)
        assert float(with_cone.image(rb).sum()) <= float(without.image(rb).sum()) + 1e-9

    def test_show_sensor_chain_smoke(self):
        pytest.importorskip("trimesh")
        from iactrace import show_sensor_chain

        cam, _ = _hex_camera(with_cone=True)
        scene = show_sensor_chain(cam)
        assert len(scene.geometry) >= 3


# 6. Alignment with a rotated sensor layout


def _rot2(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(pts) @ np.array([[c, -s], [s, c]]).T


class TestRotationAlignment:
    def _setup(self):
        size = 0.02
        centers0 = np.asarray(make_hex_centers(2, size))
        inrad = float(HexagonalSensorGroup([[0, 0, 0]], [[0, 0, 0]], centers0).hex_inradius)
        cone = OkumuraCone.quadratic(
            6,
            inrad,
            np.sin(np.radians(20.0)) * inrad,
            QUAD_P1,
            reflectivity=1.0,
            max_bounces=60,
        )
        rng = np.random.default_rng(1)
        n = 4000
        r = 0.8 * inrad * np.sqrt(rng.uniform(0, 1, n))
        psi = rng.uniform(0, 2 * np.pi, n)
        xy0 = np.c_[r * np.cos(psi), r * np.sin(psi)]
        al = np.radians(21.0)
        d0 = np.tile([np.sin(al), 0.0, -np.cos(al)], (n, 1))
        o0 = np.c_[xy0, np.full(n, 0.05)]
        return centers0, cone, o0, d0, n

    def _transmission(self, centers0, cone, o0, d0, n, theta):
        sensor = HexagonalSensorGroup(
            [[0, 0, 0]],
            [[0, 0, 0]],
            _rot2(centers0, theta),
            concentrator=cone,
            photosensor=UniformQE(1.0),
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

    def test_transmission_invariant_under_layout_rotation(self):
        centers0, cone, o0, d0, n = self._setup()
        base = self._transmission(centers0, cone, o0, d0, n, 0.0)
        assert 0.1 * n < base < 0.9 * n
        for deg in (5.0, 11.0, 23.0, 37.0):
            rotated = self._transmission(centers0, cone, o0, d0, n, np.radians(deg))
            assert abs(rotated - base) < 1e-6 * n

    def test_orientation_offset_changes_transmission(self):
        centers0, cone, o0, d0, n = self._setup()
        offset = OkumuraCone.quadratic(
            6,
            cone.entrance_apothem,
            cone.exit_apothem,
            QUAD_P1,
            reflectivity=1.0,
            max_bounces=60,
            orientation_deg=30.0,
        )
        base = self._transmission(centers0, cone, o0, d0, n, 0.0)
        shifted = self._transmission(centers0, offset, o0, d0, n, 0.0)
        assert abs(shifted - base) > 0.02 * n


# 7. Numerical robustness


class TestRobustness:
    def test_apply_gradients_are_finite(self):
        cone = OkumuraCone.cubic(
            6, 0.025, 0.010, CUBIC_P1, CUBIC_P2, reflectivity=0.9, max_bounces=8
        )

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
        cone = OkumuraCone.quadratic(6, 1.25e-4, 5e-5, QUAD_P1, reflectivity=0.95, max_bounces=12)
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
