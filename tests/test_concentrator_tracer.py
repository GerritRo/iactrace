import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import OkumuraCone, TabulatedResponse, WinstonCone
from iactrace.camera import trace_chain
from iactrace.camera.detector import DetectionSurface
from iactrace.core.ray_bundle import RayBundle

# Okumura (2012) Table 1 quadratic control point (rho1=20 mm, rho2=10 mm).
QUAD_P1 = (0.89, 0.35)

CONE_KINDS = ["winston", "okumura"]


def _make_cone(kind, exit_apothem=0.01, cutoff_deg=25.0, **kwargs):
    """A hexagonal cone of the given kind with identical (a1, a2) dimensions."""
    a1 = exit_apothem / math.sin(math.radians(cutoff_deg))
    if kind == "winston":
        return WinstonCone(6, a1, exit_apothem, **kwargs)
    return OkumuraCone.quadratic(6, a1, exit_apothem, QUAD_P1, **kwargs)


@pytest.fixture(params=CONE_KINDS)
def cone_kind(request):
    return request.param


def _fill_entrance(cone, n, seed=0):
    """Random points uniformly inside the cone entrance polygon."""
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


def _mouth_rays(cone, alpha_deg=0.0, n=1000, seed=3):
    """(xy, dirs) for ``n`` rays entering strictly inside the mouth at incidence
    ``alpha_deg``, so the tracer's mouth mask never triggers and every ray is
    genuinely traced."""
    rng = np.random.default_rng(seed)
    xy = _fill_entrance(cone, n, seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    a = math.radians(alpha_deg)
    dirs = np.stack(
        [np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.full(n, -np.cos(a))],
        axis=1,
    )
    return jnp.asarray(xy), jnp.asarray(dirs)


def _entrance_rays(xy, dirs):
    """Bundle of unit rays entering the mouth plane ``z = 0`` (pixel-local frame)."""
    n = xy.shape[0]
    return RayBundle(
        origins=jnp.concatenate([xy, jnp.zeros((n, 1))], axis=1),
        directions=dirs,
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


class TestWallsProvider:
    def test_cone_is_its_own_wall_provider(self):
        # The cone is passed straight to trace_chain: it carries the per-bounce
        # dimensions and the (M, 2) facet plane normals the tracer consumes.
        cone = _make_cone("winston", reflectivity=0.8, max_bounces=9)
        assert cone.reflectivity == pytest.approx(0.8)
        assert cone.max_bounces == 9
        assert np.asarray(cone.n_hats).shape == (cone.n_sides, 2)


class TestTraceChain:
    def test_flat_exit_stop_matches_apply(self, cone_kind):
        # A flat plane stop at the exit reproduces Concentrator.apply ray by
        # ray: apply() is the same shared tracer with a flat exit stop. Checked
        # at normal and oblique incidence (different bounce counts).
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=14)
        for alpha in (0.0, 20.0):
            rays = _entrance_rays(*_mouth_rays(cone, alpha))
            old = cone.apply(rays)
            new = trace_chain(
                cone,
                DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4),
                rays,
                max_bounces=14,
            )
            assert jnp.allclose(old.values, new.rays.values, atol=1e-6)
            # Every in-mouth ray that apply() transmits lands on the full stop.
            transmitted = np.asarray(old.values) > 0
            assert np.all(np.asarray(new.rays.alive)[transmitted])

    def test_stop_below_exit_receives_rays(self, cone_kind):
        # A photocathode in the gap below the exit still collects; landings sit
        # below the exit plane (z < -length), reached by free flight.
        cone = _make_cone(cone_kind, reflectivity=0.95, max_bounces=16)
        a2 = cone.exit_apothem
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=1.1 * a2
        )
        tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, 0.0)))
        landed = np.asarray(tr.rays.alive)
        assert landed.mean() > 0.4
        land_z = np.asarray(tr.rays.origins[:, 2])[landed]
        assert np.all(land_z < -cone.length)

    def test_peeking_dome_lands_inside_cavity(self, cone_kind):
        # A dome whose apex peeks above the exit plane is hit *inside* the
        # cavity: some landings have z > -length -- the whole point of "peeking".
        cone = _make_cone(cone_kind, reflectivity=0.95, max_bounces=16)
        a2 = cone.exit_apothem
        stop = DetectionSurface(
            vertex_z=-cone.length + 0.4 * a2, curvature=-1.0 / (1.4 * a2), radius=0.9 * a2
        )
        tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, 0.0)))
        landed = np.asarray(tr.rays.alive)
        assert landed.sum() > 0
        land_z = np.asarray(tr.rays.origins[:, 2])[landed]
        assert np.any(land_z > -cone.length), "peeking dome should catch rays inside the cavity"

    def test_wall_bounces_clamped_to_cavity(self, cone_kind):
        # With the stop below the exit, no wall bounce may sit below it: the
        # only sub-exit points are terminal landings. Otherwise the extended
        # wall surface would hand back a spurious root below the exit.
        cone = _make_cone(cone_kind, reflectivity=0.95, max_bounces=16)
        a2, length = cone.exit_apothem, cone.length
        stop = DetectionSurface(vertex_z=-length - 0.3 * a2, curvature=0.0, radius=1e4)
        tr = trace_chain(
            cone,
            stop,
            _entrance_rays(*_mouth_rays(cone, 18.0)),
            record_trajectory=True,
        )
        traj = np.asarray(tr.trajectory)  # (steps+1, N, 3)
        final_z = traj[-1, :, 2]
        # No vertex ever rises above the mouth plane (z = 0).
        assert np.all(traj[:, :, 2] <= 1e-4 * length)
        # Any vertex below the exit (z = -length) must be that ray's terminal landing.
        below = traj[:, :, 2] < -length - 1e-4 * a2
        at_landing = np.abs(traj[:, :, 2] - final_z[None, :]) < 1e-4 * a2
        assert np.all(~below | at_landing), "a wall bounce leaked below the exit plane"

    def test_aperture_bounds_and_monotonicity(self):
        # A PMT has a limited diameter: no ray lands outside the DetectionSurface
        # aperture radius, and a smaller aperture collects strictly fewer rays.
        # Aperture clipping is wall-shape-independent -> one representative cone.
        cone = _make_cone("winston", reflectivity=0.95, max_bounces=16)
        a2 = cone.exit_apothem
        rays = _entrance_rays(*_mouth_rays(cone, 12.0, n=1200))

        def collect(radius_frac):
            stop = DetectionSurface(
                vertex_z=-cone.length - 0.2 * a2,
                curvature=-1.0 / (1.2 * a2),
                radius=radius_frac * a2,
            )
            tr = trace_chain(cone, stop, rays)
            pts = np.asarray(tr.rays.origins)[np.asarray(tr.rays.alive)]
            r = np.hypot(pts[:, 0], pts[:, 1]) if len(pts) else np.array([0.0])
            return float(np.mean(tr.rays.alive)), r.max()

        for frac in (0.4, 0.7, 1.1):
            _, rmax = collect(frac)
            assert rmax <= frac * a2 + 1e-6  # nothing lands outside the aperture
        assert collect(0.4)[0] < collect(0.7)[0] < collect(1.1)[0]  # smaller collects less

    def test_rays_conserved_and_bounded(self):
        # Energy conservation is wall-shape-independent -> one representative cone.
        cone = _make_cone("winston", reflectivity=0.9, max_bounces=16)
        a2 = cone.exit_apothem
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=1.1 * a2
        )
        for alpha in (0.0, 30.0):
            tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, alpha)))
            v = np.asarray(tr.rays.values)
            assert not np.isnan(v).any()
            assert np.all(v <= 1.0 + 1e-6) and np.all(v >= 0.0)
            # dead rays carry zero throughput (the RayBundle invariant)
            assert np.all(v[~np.asarray(tr.rays.alive)] == 0.0)

    def test_trace_chain_is_jittable(self):
        cone = _make_cone("winston", reflectivity=0.9, max_bounces=16)
        a2 = cone.exit_apothem
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=1.1 * a2
        )
        rays = _entrance_rays(*_mouth_rays(cone, 5.0))
        fn = jax.jit(lambda r: trace_chain(cone, stop, r, max_bounces=16).rays.values)
        assert np.isfinite(np.asarray(fn(rays))).all()


def _rays_at(cone, alpha_deg, wavelength, n=2000, seed=3):
    """Mouth rays entering *cone* at incidence ``alpha_deg`` and one wavelength."""
    xy, dirs = _mouth_rays(cone, alpha_deg, n=n, seed=seed)
    m = xy.shape[0]
    return RayBundle(
        origins=jnp.concatenate([xy, jnp.zeros((m, 1))], axis=1),
        directions=dirs,
        values=jnp.ones(m),
        path_length=jnp.zeros(m),
        n=jnp.ones(m),
        wavelength=jnp.full(m, float(wavelength)),
    )


# A wavelength-only reflectivity curve
def _refl_curve():
    return TabulatedResponse.from_degrees(
        angles_deg=[0.0, 90.0],
        values=[[0.5, 1.0], [0.5, 1.0]],
        n_elements=1,
        wavelengths=[300.0, 500.0],
    )


# An angle-only reflectivity curve, rising from normal incidence toward grazing.
_ANGLES_DEG = [0.0, 30.0, 60.0, 90.0]
_ANGLE_VALUES = [0.2, 0.5, 0.75, 1.0]


def _angle_curve():
    return TabulatedResponse.from_degrees(
        angles_deg=_ANGLES_DEG, values=_ANGLE_VALUES, n_elements=1
    )


def _flat_curve(value):
    """An angle-flat curve pinned to ``value`` (what cos = 1 alone would give)."""
    return TabulatedResponse.from_degrees(
        angles_deg=[0.0, 90.0], values=[value, value], n_elements=1
    )


def _ones(x):
    return jnp.ones_like(jnp.asarray(x, dtype=float))


class TestWavelengthReflectivity:
    def test_uncoated_cone_is_wavelength_flat(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.85)
        assert cone.reflectivity_curve is None
        wl = jnp.array([250.0, 400.0, 650.0])
        r = np.asarray(cone.wall_reflectivity(_ones(wl), wl))
        assert np.allclose(r, 0.85)

    def test_wall_reflectivity_follows_curve(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.9, reflectivity_curve=_refl_curve())
        wl = jnp.array([300.0, 400.0, 500.0])
        r = np.asarray(cone.wall_reflectivity(_ones(wl), wl))
        assert np.allclose(r, 0.9 * np.array([0.5, 0.75, 1.0]), atol=1e-5)
        # This curve is angle-flat, so an oblique incidence gives the same values.
        oblique = np.asarray(cone.wall_reflectivity(jnp.full(wl.shape, 0.3), wl))
        assert np.allclose(oblique, r, atol=1e-5)

    def test_throughput_matches_reflectivity_power(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=16,
                          reflectivity_curve=_refl_curve())
        stop = DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4)

        def deliver(wl):
            tr = trace_chain(cone, stop, _rays_at(cone, 20.0, wl))
            return (
                np.asarray(tr.rays.values),
                np.asarray(tr.bounces),
                np.asarray(tr.rays.alive),
            )

        v_lo, b_lo, alive = deliver(300.0)  # multiplier 0.5 -> R = 0.45
        v_hi, b_hi, _ = deliver(500.0)  # multiplier 1.0 -> R = 0.90
        assert np.array_equal(b_lo, b_hi)
        # Per landed ray, the throughput is the reflectivity raised to the count
        for b in np.unique(b_hi[alive]):
            m = alive & (b_hi == b)
            assert np.allclose(v_lo[m], 0.45**b, atol=1e-5)
            assert np.allclose(v_hi[m], 0.90**b, atol=1e-5)
        # Higher reflectivity delivers strictly more light overall
        assert v_hi[alive].sum() > v_lo[alive].sum()
        z0 = alive & (b_hi == 0)
        assert np.allclose(v_lo[z0], v_hi[z0])

    def test_constant_curve_equals_scaled_scalar(self, cone_kind):
        # A flat ConstantResponse multiplier is exactly equivalent to folding that
        # factor into the scalar reflectivity.
        curve = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0], values=[0.8, 0.8], n_elements=1
        )
        coated = _make_cone(cone_kind, reflectivity=0.9, max_bounces=16,
                            reflectivity_curve=curve)
        scalar = _make_cone(cone_kind, reflectivity=0.9 * 0.8, max_bounces=16)
        stop = DetectionSurface(vertex_z=-coated.length, curvature=0.0, radius=1e4)
        rays = _rays_at(coated, 18.0, 450.0)
        vc = np.asarray(trace_chain(coated, stop, rays).rays.values)
        vs = np.asarray(trace_chain(scalar, stop, rays).rays.values)
        assert np.allclose(vc, vs, atol=1e-6)

    def test_wavelength_reflectivity_is_differentiable(self):
        # grad of delivered light w.r.t. the reflectivity curve value flows
        # through the bounce loop.
        cone = _make_cone("winston", reflectivity=0.9, max_bounces=16)
        stop = DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4)
        rays = _rays_at(cone, 20.0, 400.0)

        def total(mult):
            curve = TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0],
                values=[[mult, mult], [mult, mult]],
                n_elements=1,
                wavelengths=[300.0, 500.0],
            )
            coned = WinstonCone(
                cone.n_sides, cone.entrance_apothem, cone.exit_apothem,
                length=cone.length, reflectivity=0.9, max_bounces=16,
                reflectivity_curve=curve,
            )
            return trace_chain(coned, stop, rays).rays.values.sum()

        g = jax.grad(total)(0.7)
        assert np.isfinite(float(g))
        # More reflective walls can only deliver more light
        assert float(g) > 0.0


class TestAngularReflectivity:
    def test_wall_reflectivity_follows_angle(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.9, reflectivity_curve=_angle_curve())
        cos = jnp.cos(jnp.deg2rad(jnp.asarray(_ANGLES_DEG)))
        r = np.asarray(cone.wall_reflectivity(cos, jnp.full(cos.shape, 400.0)))
        assert np.allclose(r, 0.9 * np.asarray(_ANGLE_VALUES), atol=1e-6)
        # Between the tabulated nodes the curve stays monotone toward grazing.
        cos_fine = jnp.cos(jnp.deg2rad(jnp.linspace(0.0, 90.0, 40)))
        r_fine = np.asarray(cone.wall_reflectivity(cos_fine, jnp.full(cos_fine.shape, 400.0)))
        assert np.all(np.diff(r_fine) > 0.0)

    def test_bounce_uses_actual_incidence_angle(self, cone_kind):
        # For a ray that bounces exactly once, the delivered throughput must be
        # the coating evaluated at *that bounce's* incidence angle. The angle is
        # recovered from the trajectory without touching the tracer's own
        # bookkeeping: d_in . d_out = 1 - 2 cos^2(theta_i) for a specular bounce.
        cone = _make_cone(
            cone_kind, reflectivity=0.9, max_bounces=16, reflectivity_curve=_angle_curve()
        )
        stop = DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4)
        rays = _rays_at(cone, 15.0, 400.0)
        tr = trace_chain(cone, stop, rays, record_trajectory=True)

        traj = np.asarray(tr.trajectory)  # (steps+1, N, 3)
        sel = np.asarray(tr.rays.alive) & (np.asarray(tr.bounces) == 1)
        assert sel.sum() > 20, "need a decent sample of single-bounce rays"

        d_in = np.asarray(rays.directions)[sel]
        seg = traj[2, sel] - traj[1, sel]  # bounce point -> landing point
        d_out = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        cos_i = np.sqrt(np.clip(0.5 * (1.0 - np.sum(d_in * d_out, axis=1)), 0.0, 1.0))
        assert cos_i.max() < 0.99, "these bounces should be genuinely off-normal"

        expected = np.asarray(
            cone.wall_reflectivity(jnp.asarray(cos_i), jnp.full(cos_i.shape, 400.0))
        )
        assert np.allclose(np.asarray(tr.rays.values)[sel], expected, atol=1e-4)

    def test_angle_curve_beats_its_normal_incidence_value(self, cone_kind):
        # Regression: the walls used to evaluate the coating at cos = 1 for every
        # bounce, which is exactly an angle-flat curve pinned to the
        # normal-incidence value. With R rising toward grazing, every bounced ray
        # must now come out strictly brighter than that.
        angled = _make_cone(
            cone_kind, reflectivity=0.9, max_bounces=16, reflectivity_curve=_angle_curve()
        )
        pinned = _make_cone(
            cone_kind,
            reflectivity=0.9,
            max_bounces=16,
            reflectivity_curve=_flat_curve(_ANGLE_VALUES[0]),
        )
        stop = DetectionSurface(vertex_z=-angled.length, curvature=0.0, radius=1e4)
        rays = _rays_at(angled, 15.0, 400.0)

        tr_a = trace_chain(angled, stop, rays)
        tr_p = trace_chain(pinned, stop, rays)
        # Reflectivity never moves a ray, so both runs share the same geometry.
        assert np.array_equal(np.asarray(tr_a.bounces), np.asarray(tr_p.bounces))

        alive = np.asarray(tr_a.rays.alive)
        bounces = np.asarray(tr_a.bounces)
        v_a, v_p = np.asarray(tr_a.rays.values), np.asarray(tr_p.rays.values)
        straight, bounced = alive & (bounces == 0), alive & (bounces > 0)
        assert bounced.sum() > 0
        assert np.allclose(v_a[straight], v_p[straight])  # no bounce, no coating
        assert np.all(v_a[bounced] > v_p[bounced] + 1e-6)

    def test_angle_reflectivity_is_differentiable(self, cone_kind):
        # grad of delivered light w.r.t. the grazing end of the R(theta) curve
        # flows through the per-bounce coating lookup.
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=16)
        stop = DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4)
        rays = _rays_at(cone, 20.0, 400.0)

        def total(grazing):
            curve = TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0], values=jnp.stack([jnp.asarray(0.2), grazing]), n_elements=1
            )
            coned = _make_cone(
                cone_kind, reflectivity=0.9, max_bounces=16, reflectivity_curve=curve
            )
            return trace_chain(coned, stop, rays).rays.values.sum()

        g = jax.grad(total)(0.8)
        assert np.isfinite(float(g))
        assert float(g) > 0.0  # more reflective at grazing -> more light delivered
