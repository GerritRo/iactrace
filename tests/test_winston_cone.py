import math

import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Camera, ConstantQE, HexagonalSensorGroup, WinstonCone
from iactrace.camera.winston_cone import cpc_full_length, cpc_wall_tilt, profile_apothem
from iactrace.core.ray_bundle import RayBundle

from .test_sensors import make_hex_centers


def _cone(exit_apothem, cutoff_deg, n_sides=6, **kwargs):
    """Build a full cone with the given exit apothem and design cutoff (test helper)."""
    a1 = exit_apothem / math.sin(math.radians(cutoff_deg))
    return WinstonCone(n_sides, a1, exit_apothem, **kwargs)


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


def _transmission(cone, alpha_deg, n=3000):
    return float((np.asarray(_launch(cone, alpha_deg, n).values) > 0).mean())


# 1. Geometry


class TestGeometry:
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
        assert float(r[0]) == pytest.approx(a2, abs=1e-9)  # exit
        assert float(r[-1]) == pytest.approx(a1, abs=1e-6)  # entrance
        assert bool(jnp.all(jnp.diff(r) >= -1e-9))  # monotonic

    def test_concentration_ratio(self):
        a2 = 0.5
        s = math.sin(math.radians(20.0))
        a1 = a2 / s
        assert (a1 / a2) ** 2 == pytest.approx(1.0 / s**2)

    def test_constructors_and_truncation(self):
        a2 = 0.5
        s_d, c_d = math.sin(math.radians(20.0)), math.cos(math.radians(20.0))
        a1 = a2 / s_d  # full mouth
        length = cpc_full_length(a2, s_d, c_d)

        full = WinstonCone(6, a1, a2)  # entrance, exit (full)
        assert full.length == pytest.approx(length)
        assert full.entrance_apothem == pytest.approx(a1, abs=1e-6)
        assert full._wall_tilt()[0] == pytest.approx(s_d, abs=1e-6)

        # Truncated cone given by its PHYSICAL mouth at z = length: the wall tilt
        # (hence the parabola) is recovered exactly from (a2, a1, length).
        phys = float(profile_apothem(jnp.asarray(2.0), a2, s_d, c_d))
        assert phys < a1  # truncated mouth is smaller
        trunc = WinstonCone(6, phys, a2, length=2.0)
        assert trunc.length == pytest.approx(2.0)
        assert trunc.entrance_apothem == pytest.approx(phys)  # mouth == what we passed
        assert trunc._wall_tilt()[0] == pytest.approx(s_d, abs=1e-6)  # same parabola

        with pytest.raises(ValueError):
            WinstonCone(6, 0.5, 0.6)  # exit > entrance
        with pytest.raises(ValueError):
            # mouth far too wide for this shallow depth -> not a realizable CPC
            WinstonCone(6, a1, a2, length=0.05)


# 2. Angular acceptance (the defining CPC property)


class TestAcceptance:
    def test_cutoff_near_acceptance_angle(self):
        cone = _cone(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        # well inside -> almost everything transmits
        assert _transmission(cone, 0.0) > 0.95
        assert _transmission(cone, 12.0) > 0.9
        # around the design angle -> roughly half
        assert 0.35 < _transmission(cone, 20.0) < 0.8
        # well outside -> strongly rejected
        assert _transmission(cone, 28.0) < 0.15
        assert _transmission(cone, 40.0) < 0.02

    def test_monotone_decreasing(self):
        cone = _cone(0.5, 25.0, reflectivity=1.0, max_bounces=40)
        ts = [_transmission(cone, a) for a in (0, 10, 20, 25, 30, 40)]
        assert all(ts[i] >= ts[i + 1] - 0.08 for i in range(len(ts) - 1))


# 3. Energy conservation


class TestEnergy:
    def test_no_gain_and_attenuated(self):
        cone = _cone(0.5, 20.0, reflectivity=0.9, max_bounces=40)
        out = _launch(cone, 8.0, n=4000)
        v = np.asarray(out.values)
        assert v.max() <= 1.0 + 1e-9  # never amplifies
        assert not bool((v > 1.0 + 1e-9).any())
        transmitted = v[v > 0]
        assert transmitted.min() >= 0.9**40 - 1e-12  # >= reflectivity^max_bounces
        assert transmitted.mean() < 1.0  # some bounces happened

    def test_reflectivity_one_keeps_unit_values(self):
        cone = _cone(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        out = _launch(cone, 5.0, n=2000)
        v = np.asarray(out.values)
        assert set(np.unique(np.round(v, 9)).tolist()) <= {0.0, 1.0}

    def test_exit_plane_and_path_length(self):
        cone = _cone(0.5, 20.0, reflectivity=1.0, max_bounces=40)
        out = _launch(cone, 6.0, n=1500)
        assert bool(np.allclose(np.asarray(out.origins)[:, 2], -cone.length))
        v = np.asarray(out.values)
        # transmitted rays travel at least the cone depth
        assert np.asarray(out.path_length)[v > 0].min() >= cone.length - 1e-9


# 4. Pure tracer + jit


class TestTracer:
    def test_rays_outside_mouth_are_lost(self):
        cone = _cone(0.5, 20.0, reflectivity=1.0)
        a = cone.entrance_apothem
        # one ray just inside the mouth, one far outside
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

    def test_jit(self):
        import jax

        cone = _cone(0.5, 20.0)
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
        WinstonCone(
            6,
            entrance_apothem=entrance,
            exit_apothem=0.35 * entrance,
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
        photosensor=ConstantQE(0.9),
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
        rb = _downward(np.zeros((50, 2)))  # on-axis rays into centre pixel
        img = cam.image(rb)
        assert img.shape == (1, sensor.n_pixels)
        assert float(img.sum()) > 0.0
        pe, t, pix, hit = cam.collect(rb)
        assert pe.shape == (50,) and t.shape == (50,)
        assert bool((np.asarray(pe) <= 0.9 + 1e-9).all())  # <= QE, attenuated

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
        assert len(scene.geometry) >= 3  # entrance + faceted cone + detector


# 6. Alignment with a rotated sensor layout


def _rot2(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(pts) @ np.array([[c, -s], [s, c]]).T


class TestRotationAlignment:
    """The cone must stay aligned with its pixel under any hex grid rotation.

    ``to_pixel_frame`` normalises ``grid_rotation`` away, so the cone (defined in
    the grid-aligned pixel frame) co-rotates with the layout.
    """

    def _setup(self):
        size = 0.02
        centers0 = np.asarray(make_hex_centers(2, size))
        inrad = float(HexagonalSensorGroup([[0, 0, 0]], [[0, 0, 0]], centers0).hex_inradius)
        cone = WinstonCone(
            6,
            entrance_apothem=inrad,
            exit_apothem=np.sin(np.radians(20.0)) * inrad,
            reflectivity=1.0,
            max_bounces=60,
        )
        rng = np.random.default_rng(1)
        n = 4000
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
            photosensor=ConstantQE(1.0),
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
        assert 0.1 * n < base < 0.9 * n  # genuinely partial -> discriminating
        for deg in (5.0, 11.0, 23.0, 37.0, 50.0):
            rotated = self._transmission(centers0, cone, o0, d0, n, np.radians(deg))
            assert abs(rotated - base) < 1e-6 * n

    def test_orientation_offset_changes_transmission(self):
        # Control: the test is sensitive -- offsetting the cone orientation
        # relative to the pixel changes the result (so invariance above is real).
        centers0, cone, o0, d0, n = self._setup()
        offset = WinstonCone(
            6,
            entrance_apothem=cone.entrance_apothem,
            exit_apothem=cone.exit_apothem,
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
        # The trace's divisions/sqrts must be grad-safe (double-where), including
        # the degenerate horizontal ray (d[2] == 0) that hits the inf branch.
        import jax

        cone = WinstonCone(
            6, entrance_apothem=0.025, exit_apothem=0.01, reflectivity=0.9, max_bounces=8
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
        # The accept/reject floor scales with a2, so a sub-mm cone still
        # transmits finite values in (0, 1] rather than absorbing everything.
        cone = WinstonCone(
            6, entrance_apothem=1.25e-4, exit_apothem=5e-5, reflectivity=0.95, max_bounces=12
        )
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
