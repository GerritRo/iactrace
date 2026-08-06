"""float32 / float64 agreement across the traced path.

The package runs in float32 (``conftest.py`` pins ``jax_enable_x64`` off), so
every tolerance in the traced kernels is derived from the active float dtype
rather than hard-coded -- see :mod:`iactrace.core._tolerances`. The property
that buys is checked here directly: trace the same geometry twice, once in
float32 and once in float64, and require the two to agree. Tolerances that
track the dtype tighten with it and the runs converge; tolerances fixed at some
absolute value do not, and the float32 run diverges or drops rays the float64
run keeps.

The whole-telescope cases pin the mainstream path. The kernel cases below them
cover the geometries a fixed threshold gets wrong -- a near-on-axis paraboloid,
a steep asphere, grazing incidence under total internal reflection -- which are
rare enough in a random ray fan not to show up in the telescope traces at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# eps of the dtype the traced path actually runs in.
EPS32 = float(np.finfo(np.float32).eps)


@pytest.fixture(autouse=True)
def _allow_x64(set_jax_precision):
    """Let a test in this module switch x64 on, and put it back afterwards.

    ``conftest.set_jax_precision`` is autouse and forces x64 *off* before every
    test; depending on it here orders this fixture after it, so the restore
    below is the last word.
    """
    yield
    jax.config.update("jax_enable_x64", False)


def _traced_in(dtype64, build):
    """Run ``build`` with x64 set to ``dtype64``, returned as float64 arrays.

    Everything the comparison touches has to be *created* inside the enabled
    region: flipping ``jax_enable_x64`` does not retype arrays that already
    exist, so a telescope built beforehand would stay float32 whatever the flag
    says.
    """
    jax.config.update("jax_enable_x64", dtype64)
    try:
        return jax.tree_util.tree_map(lambda a: np.asarray(a, dtype=np.float64), build())
    finally:
        jax.config.update("jax_enable_x64", False)


# --- whole-telescope traces ----------------------------------------------------


def _trace_config(telescope_yaml, camera_yaml, aperture_radius, height, n_rays=400):
    """Spot positions and detection mask for a fan of rays down the optic axis."""
    from iactrace import Camera, Telescope
    from iactrace.camera.camera import intersect_sensor

    telescope = Telescope.from_yaml(telescope_yaml, 8, key=jax.random.key(42))
    camera = Camera.from_yaml(camera_yaml)

    rng = np.random.default_rng(7)
    r = aperture_radius * np.sqrt(rng.uniform(size=n_rays))
    phi = rng.uniform(0.0, 2.0 * np.pi, n_rays)
    origins = jnp.asarray(
        np.stack([r * np.cos(phi), r * np.sin(phi), np.full(n_rays, height)], axis=-1)
    )
    directions = jnp.broadcast_to(jnp.asarray([0.0, 0.0, -1.0]), (n_rays, 3))

    rays = telescope.trace(origins, directions, jnp.ones(n_rays)).rays
    sensor_rays, _ = intersect_sensor(camera, rays)
    return sensor_rays.origins, rays.values > 0


class TestTelescopePrecision:
    """A full trace lands in the same place whichever float dtype it runs in."""

    # (telescope, camera, aperture radius, source height) spanning the range of
    # scales the optics have to work at: a 0.1 m bench mirror to a 28 m dish.
    CONFIGS = [
        ("configs/BASIC/Parabolic.yaml", "configs/BASIC/default_camera.yaml", 0.09, 5.0),
        ("configs/BASIC/Cassegrain.yaml", "configs/BASIC/default_camera.yaml", 0.09, 5.0),
        ("configs/CTAO/LST_1_North_like.yaml", "configs/CTAO/LSTCam.yaml", 10.0, 100.0),
        ("configs/HESS/CT3.yaml", "configs/HESS/HESS1U.yaml", 5.0, 100.0),
    ]

    @pytest.mark.parametrize(("telescope", "camera", "radius", "height"), CONFIGS)
    def test_spot_positions_agree(self, telescope, camera, radius, height):
        args = (telescope, camera, radius, height)
        xy32, hit32 = _traced_in(False, lambda: _trace_config(*args))
        xy64, hit64 = _traced_in(True, lambda: _trace_config(*args))

        # Same rays detected. This is the half that a too-tight residual bound
        # breaks: the float32 run reports converged hits as misses and silently
        # drops them, while float64 keeps them.
        assert np.array_equal(hit32, hit64), (
            f"{int((hit32 != hit64).sum())} rays detected in one dtype but not the other"
        )

        detected = hit32.astype(bool)
        assert detected.sum() > 0.5 * len(detected), "geometry does not illuminate the camera"

        # Landing positions agree to a few float32 ulps at this telescope's own
        # scale -- far below a pixel, and far below the PSF.
        deviation = np.linalg.norm(xy32[detected] - xy64[detected], axis=-1)
        assert deviation.max() <= 8.0 * EPS32 * height, (
            f"max deviation {deviation.max():.3e} m exceeds "
            f"{8.0 * EPS32 * height:.3e} m at a source height of {height} m"
        )


# --- intersection kernels ------------------------------------------------------


def _near_axis_paraboloid():
    """Ray parameters for a fan sweeping through the axis of a paraboloid.

    The on-axis limit is where the quadratic's leading coefficient goes to zero:
    ``A = c (dx^2 + dy^2 + (1 + k) dz^2)`` vanishes with the off-axis angle for
    ``k = -1``, so the textbook root formula loses the intersection to
    cancellation exactly where the rays matter most.
    """
    from iactrace.core.intersections import intersect_conic

    angles = np.linspace(0.0, 5e-3, 2000)
    directions = jnp.asarray(
        np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=-1)
    )
    origins = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 20.0]), directions.shape)
    solve = jax.jit(jax.vmap(lambda o, d: intersect_conic(o, d, 0.05, -1.0)))
    return solve(origins, directions)


def _steep_asphere(n_rays=3000):
    """Newton intersections on a surface whose sag is metres, not millimetres.

    The residual ``z - sag(x, y)`` is then large enough that its float32
    round-off exceeds any fixed tolerance tight enough to be meaningful
    elsewhere, which is what makes converged hits look like misses.
    """
    from iactrace.core.intersections import newton_raphson_intersect

    def sag(x, y):
        r2 = x * x + y * y
        return 0.05 * r2 / 2 + 1e-4 * r2 * r2

    rng = np.random.default_rng(0)
    x0, y0 = rng.uniform(-1.5, 1.5, (2, n_rays))
    theta = rng.uniform(0.0, 1.3, n_rays)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_rays)
    directions = jnp.asarray(
        np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], axis=-1
        )
    )
    origins = jnp.asarray(np.stack([x0, y0, np.full(n_rays, 20.0)], axis=-1))
    solve = jax.jit(jax.vmap(lambda o, d: newton_raphson_intersect(sag, o, d)))
    t, _, valid = solve(origins, directions)
    return t, valid


class TestKernelPrecision:
    """The degenerate geometries, where a fixed threshold is furthest off."""

    def test_near_axis_paraboloid_agrees(self):
        t32 = _traced_in(False, _near_axis_paraboloid)
        t64 = _traced_in(True, _near_axis_paraboloid)

        finite = np.isfinite(t32) & np.isfinite(t64)
        assert finite.all(), "every ray in this fan crosses the paraboloid"

        # 20 m of travel in float32 carries ~2e-6 m of round-off; anything much
        # above that is the cancelling root formula, not the dtype.
        deviation = np.abs(t32[finite] - t64[finite])
        assert deviation.max() <= 8.0 * EPS32 * 20.0, (
            f"max deviation {deviation.max():.3e} m: the on-axis root has lost "
            "its significance to cancellation"
        )

    def test_steep_asphere_accepts_the_same_rays(self):
        t32, valid32 = _traced_in(False, _steep_asphere)
        t64, valid64 = _traced_in(True, _steep_asphere)

        valid32, valid64 = valid32.astype(bool), valid64.astype(bool)
        assert valid64.all(), "every ray here does hit the surface"
        assert np.array_equal(valid32, valid64), (
            f"{int((valid32 != valid64).sum())} converged hits rejected in float32 "
            "but accepted in float64 -- the residual bound is not tracking the dtype"
        )

        deviation = np.abs(t32[valid32] - t64[valid32])
        assert deviation.max() <= 8.0 * EPS32 * 20.0

    def test_fresnel_grazing_under_tir(self):
        """At grazing incidence under TIR the limit is R = 1 in either dtype.

        Both denominators vanish there, and so do both numerators. Clamping the
        denominator turns that 0/0 into 0 and transmits a ray that should have
        been reflected outright.
        """
        from iactrace.core.coatings import fresnel_unpolarized

        def grazing():
            cos_i = jnp.asarray(np.array([0.0, 1e-9, 1e-7, 1e-5, 1e-3]))
            return fresnel_unpolarized(cos_i, 1.5, 1.0)

        for x64 in (False, True):
            R, T = _traced_in(x64, grazing)
            assert np.allclose(R, 1.0), f"R = {R} at grazing incidence under TIR (x64={x64})"
            assert np.allclose(T, 0.0)
