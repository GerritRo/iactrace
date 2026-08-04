import jax
import jax.numpy as jnp

from iactrace import ConstantQE, RayBundle, TraceResult, Trajectory
from iactrace.camera import DetectionChain
from iactrace.core.render import trace_optics

from ._helpers import make_simple_telescope, make_two_stage_telescope

N_RAYS = 8


def _down_rays(n=N_RAYS, z=50.0):
    """Parallel rays heading down the optical axis, inside the mirror aperture."""
    x = jnp.linspace(-0.05, 0.05, n)
    origins = jnp.stack([x, jnp.zeros(n), jnp.full(n, z)], axis=-1)
    directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n, 3))
    return origins, directions, jnp.ones(n)


def _flat_rays(n=3):
    """Rays entering a pixel at z = 0, travelling toward the detector."""
    return RayBundle(
        origins=jnp.zeros((n, 3)),
        directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1)),
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
    )


class TestTraceOptics:
    def test_returns_result_without_trajectory_by_default(self):
        tel, _ = make_simple_telescope(n_samples=8)
        origins, directions, values = _down_rays()

        res = trace_optics(tel.optical_groups, tel.obstruction_groups, origins, directions, values)

        assert isinstance(res, TraceResult)
        assert isinstance(res.rays, RayBundle)
        assert res.trajectory is None
        # NamedTuple: still unpacks as a plain pair.
        rays, trajectory = res
        assert rays is res.rays and trajectory is None

    def test_records_one_point_per_stage_plus_source(self):
        tel, _ = make_simple_telescope(n_samples=8)
        origins, directions, values = _down_rays()

        res = trace_optics(
            tel.optical_groups,
            tel.obstruction_groups,
            origins,
            directions,
            values,
            record_trajectory=True,
        )

        assert isinstance(res.trajectory, Trajectory)
        # One optical stage: the source point, then the landing on the mirror.
        assert res.trajectory.points.shape == (2, N_RAYS, 3)
        assert jnp.allclose(res.trajectory.points[0], origins)

    def test_recording_does_not_change_the_rays(self):
        tel, _ = make_simple_telescope(n_samples=8)
        args = (tel.optical_groups, tel.obstruction_groups, *_down_rays())

        plain = trace_optics(*args)
        traced = trace_optics(*args, record_trajectory=True)

        assert jnp.allclose(plain.rays.origins, traced.rays.origins)
        assert jnp.allclose(plain.rays.values, traced.rays.values)


class TestTelescopeTrace:
    def test_default_returns_a_result_without_a_trajectory(self):
        tel, _ = make_simple_telescope(n_samples=8)

        res = tel.trace(*_down_rays())

        assert isinstance(res, TraceResult)
        assert isinstance(res.rays, RayBundle)
        assert res.trajectory is None

    def test_recording_appends_the_converging_leg(self):
        tel, _ = make_simple_telescope(n_samples=8)
        origins, directions, values = _down_rays()

        rb = tel.trace(origins, directions, values).rays
        rays, trajectory = tel.trace(origins, directions, values, record_trajectory=True)

        assert isinstance(trajectory, Trajectory)
        # source + one mirror + the landing on the camera reference plane.
        assert trajectory.points.shape == (3, N_RAYS, 3)
        assert jnp.allclose(trajectory.points[0], origins)
        # The bundle is untouched by recording, and still stops on the last optic.
        assert jnp.allclose(rays.origins, rb.origins)
        assert jnp.allclose(rays.values, rb.values)

    def test_stage_count_drives_the_step_count(self):
        tel, _ = make_two_stage_telescope(n_samples=8)
        origins, directions, values = _down_rays()

        _, trajectory = tel.trace(origins, directions, values, record_trajectory=True)

        # Two stages: source + two optics + the focal-plane landing.
        assert trajectory.points.shape == (4, N_RAYS, 3)


class TestDetectionChainPropagate:
    def test_trajectory_is_none_unless_asked(self):
        chain = DetectionChain(concentrator=None, photodetector=ConstantQE(1.0), gap=0.02)

        res = chain.propagate(_flat_rays())

        assert isinstance(res, TraceResult)
        assert res.trajectory is None
        assert isinstance(res.rays, RayBundle)

    def test_falls_back_to_a_straight_segment_without_a_concentrator(self):
        chain = DetectionChain(concentrator=None, photodetector=ConstantQE(1.0), gap=0.02)
        rays = _flat_rays()

        plain = chain.propagate(rays)
        traced = chain.propagate(rays, record_trajectory=True)

        # No cone can report an internal path: entrance straight to the landing.
        assert traced.trajectory.points.shape == (2, rays.origins.shape[0], 3)
        assert jnp.allclose(traced.trajectory.points[0], rays.origins)
        assert jnp.allclose(plain.rays.values, traced.rays.values)


class TestCameraTrace:
    def test_returns_a_result_that_always_carries_its_path(self):
        tel, cam = make_simple_telescope(n_samples=64)
        rb = tel.trace(*_down_rays()).rays

        res = cam.trace(rb)

        # Camera.trace exists to record, so unlike the other tracers its
        # trajectory is never None.
        assert isinstance(res, TraceResult)
        assert isinstance(res.trajectory, Trajectory)
        assert isinstance(res.rays, RayBundle)

    def test_path_starts_on_the_last_optic(self):
        tel, cam = make_simple_telescope(n_samples=64)
        rb = tel.trace(*_down_rays()).rays

        trajectory = cam.trace(rb).trajectory

        # The incoming leg is prepended to the chain path, so the drawn path
        # starts where the bundle entered the camera.
        assert trajectory.points.shape[0] >= 2
        assert trajectory.points.shape[1:] == (N_RAYS, 3)
        assert jnp.allclose(trajectory.points[0], rb.origins)
        assert jnp.all(jnp.isfinite(trajectory.points))

    def test_rays_match_what_collect_reports(self):
        tel, cam = make_simple_telescope(n_samples=64)
        rb = tel.trace(*_down_rays()).rays

        pe_vals, _, pix_id, detected = cam.collect(rb)
        res = cam.trace(rb)

        assert jnp.allclose(jnp.where(detected, res.rays.values, 0.0), pe_vals)


def test_trace_result_is_a_pytree():
    """The docstring promises it survives jit / scan boundaries."""
    rays = _flat_rays()
    res = TraceResult(rays)

    leaves, treedef = jax.tree_util.tree_flatten(res)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, TraceResult)
    assert rebuilt.trajectory is None
    assert jnp.allclose(rebuilt.rays.origins, rays.origins)
