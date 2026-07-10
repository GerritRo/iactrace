"""Contract tests for the shared cone tracer (:func:`iactrace.camera.trace_chain`).

Parametrized over every wall provider (Winston paraboloid, Okumura Bezier), so
each cone type gives the tracer the same guarantees: cavity-clamped wall hits,
mouth-aperture masking, correct landings on stops below the exit or peeking
into the cavity, aperture bounds, and energy conservation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import OkumuraCone, WinstonCone
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


def _mouth_rays(cone, alpha_deg=0.0, n=2000, seed=3):
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
    def test_cone_is_its_own_wall_provider(self, cone_kind):
        # The cone is passed straight to trace_chain: it carries the per-bounce
        # dimensions and the (M, 2) facet plane normals the tracer consumes.
        cone = _make_cone(cone_kind, reflectivity=0.8, max_bounces=9)
        assert cone.reflectivity == pytest.approx(0.8)
        assert cone.max_bounces == 9
        assert np.asarray(cone.n_hats).shape == (cone.n_sides, 2)


class TestTraceChain:
    @pytest.mark.parametrize("alpha", [0.0, 10.0, 20.0])
    def test_flat_exit_stop_matches_apply(self, cone_kind, alpha):
        # A flat plane stop at the exit reproduces Concentrator.apply ray by
        # ray: apply() is the same shared tracer with a flat exit stop.
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=14)
        rays = _entrance_rays(*_mouth_rays(cone, alpha))

        old = cone.apply(rays)
        new = trace_chain(
            cone,
            DetectionSurface(vertex_z=-cone.length, curvature=0.0, radius=1e4),
            rays,
            max_bounces=14,
        )

        assert jnp.allclose(old.values, new.rays.values, atol=1e-6)
        # Every in-mouth ray that apply() transmits lands on the full-aperture stop.
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

    def test_trajectory_off_by_default(self, cone_kind):
        cone = _make_cone(cone_kind)
        stop = DetectionSurface(vertex_z=-cone.length)
        tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, 5.0, n=50)))
        assert tr.trajectory is None

    @pytest.mark.parametrize("radius_frac", [0.4, 0.7, 1.1])
    @pytest.mark.parametrize("alpha", [0.0, 12.0])
    def test_aperture_bounds_landings(self, cone_kind, radius_frac, alpha):
        # A PMT has a limited diameter: no ray may land outside the
        # DetectionSurface aperture radius, at any incidence.
        cone = _make_cone(cone_kind, reflectivity=0.95, max_bounces=16)
        a2 = cone.exit_apothem
        radius = radius_frac * a2
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=radius
        )
        tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, alpha, n=1200)))
        pts = np.asarray(tr.rays.origins)[np.asarray(tr.rays.alive)]
        r = np.hypot(pts[:, 0], pts[:, 1]) if len(pts) else np.array([0.0])
        assert r.max() <= radius + 1e-6

    def test_smaller_aperture_collects_less(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.95, max_bounces=16)
        a2 = cone.exit_apothem
        rays = _entrance_rays(*_mouth_rays(cone, 0.0, n=1200))

        def coll(radius):
            stop = DetectionSurface(
                vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=radius
            )
            return float(jnp.mean(trace_chain(cone, stop, rays).rays.alive))

        assert coll(0.4 * a2) < coll(0.7 * a2) < coll(1.1 * a2)

    def test_rays_conserved_and_bounded(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=16)
        a2 = cone.exit_apothem
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=1.1 * a2
        )
        for alpha in (0.0, 15.0, 30.0):
            tr = trace_chain(cone, stop, _entrance_rays(*_mouth_rays(cone, alpha)))
            v = np.asarray(tr.rays.values)
            assert not np.isnan(v).any()
            assert np.all(v <= 1.0 + 1e-6) and np.all(v >= 0.0)
            # dead rays carry zero throughput (the RayBundle invariant)
            assert np.all(v[~np.asarray(tr.rays.alive)] == 0.0)

    def test_trace_chain_is_jittable(self, cone_kind):
        cone = _make_cone(cone_kind, reflectivity=0.9, max_bounces=16)
        a2 = cone.exit_apothem
        walls = cone
        stop = DetectionSurface(
            vertex_z=-cone.length - 0.2 * a2, curvature=-1.0 / (1.2 * a2), radius=1.1 * a2
        )
        rays = _entrance_rays(*_mouth_rays(cone, 5.0))
        fn = jax.jit(lambda r: trace_chain(walls, stop, r, max_bounces=16).rays.values)
        assert np.isfinite(np.asarray(fn(rays))).all()
