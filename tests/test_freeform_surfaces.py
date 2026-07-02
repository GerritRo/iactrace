"""Tests for the bicubic-interpolated FreeformSurfaceGroup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import (
    AsphericSurfaceGroup,
    FreeformSurfaceGroup,
    SumSurfaceGroup,
    bicubic_interp,
)


def _grid_from_fn(fn, n=21, half=0.5):
    xs = jnp.linspace(-half, half, n)
    ys = jnp.linspace(-half, half, n)
    X, Y = jnp.meshgrid(xs, ys)  # X[j,i]=xs[i], Y[j,i]=ys[j]
    return fn(X, Y), xs, ys


def _freeform(fn, n=21, half=0.5, offset=(0.0, 0.0)):
    grid, xs, ys = _grid_from_fn(fn, n, half)
    surf = FreeformSurfaceGroup.from_extent(
        grid[None], half, half, offsets=jnp.asarray([offset])
    )
    return surf, xs, ys


def _fd_slope(f, x, y, h=1e-5):
    return (
        float((f(x + h, y) - f(x - h, y)) / (2 * h)),
        float((f(x, y + h) - f(x, y - h)) / (2 * h)),
    )


class TestBicubicKernel:
    def test_node_exact(self):
        grid = jnp.array([[0.0, 1.0, 2.0, 3.0],
                          [1.0, 2.0, 3.0, 4.0],
                          [2.0, 3.0, 4.0, 5.0],
                          [3.0, 4.0, 5.0, 6.0]])
        for j in range(4):
            for i in range(4):
                assert float(bicubic_interp(grid, float(i), float(j))) == pytest.approx(
                    float(grid[j, i])
                )

    def test_clamps_out_of_range(self):
        grid = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        # far outside -> clamped to the corner value
        assert float(bicubic_interp(grid, -5.0, -5.0)) == pytest.approx(0.0)
        assert float(bicubic_interp(grid, 10.0, 10.0)) == pytest.approx(3.0)


class TestFreeformSurface:
    def test_node_exact_interpolation(self):
        def fn(x, y):
            return 0.02 * x**2 - 0.01 * y**2 + 0.005 * x * y
        surf, xs, ys = _freeform(fn)
        elem = surf._index(0)
        for i in (0, 5, 10, 20):
            for j in (0, 7, 20):
                assert float(elem._sag_intrinsic(xs[i], ys[j])) == pytest.approx(
                    float(fn(xs[i], ys[j])), abs=1e-9
                )

    def test_interpolation_accuracy(self):
        def fn(x, y):
            return 0.02 * x**2 - 0.01 * y**2 + 0.003 * jnp.sin(3 * x)
        surf, _, _ = _freeform(fn, n=41, half=0.5)
        elem = surf._index(0)
        for x, y in [(0.137, -0.21), (-0.33, 0.08), (0.4, 0.4)]:
            assert float(elem._sag_intrinsic(x, y)) == pytest.approx(
                float(fn(x, y)), abs=1e-5
            )

    def test_rezero_at_offset(self):
        def fn(x, y):
            return 0.5 + 0.02 * x**2  # non-zero at origin
        surf, _, _ = _freeform(fn)
        # sag is re-zeroed: zero at the decenter point (origin here)
        assert float(surf.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)

    def test_normal_matches_finite_difference(self):
        def fn(x, y):
            return 0.03 * x**2 - 0.02 * y**2 + 0.01 * x * y + 0.004 * jnp.sin(4 * x)
        surf, _, _ = _freeform(fn, n=41)
        elem = surf._index(0)
        for x, y in [(0.2, 0.1), (-0.3, 0.25), (0.15, -0.4)]:
            _, normal = elem.compute_sag_and_normal_at(x, y)
            dzdx, dzdy = _fd_slope(lambda xx, yy: elem._sag_local(xx, yy), x, y)
            expected = np.array([-dzdx, -dzdy, 1.0])
            expected /= np.linalg.norm(expected)
            assert np.allclose(np.asarray(normal), expected, atol=1e-4)
            assert float(jnp.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-10)

    def test_flat_grid_is_flat(self):
        surf = FreeformSurfaceGroup.from_extent(
            jnp.full((1, 9, 9), 0.7), 0.5, 0.5
        )
        elem = surf._index(0)
        for x, y in [(0.0, 0.0), (0.2, -0.3), (0.4, 0.1)]:
            assert float(elem._sag_local(x, y)) == pytest.approx(0.0, abs=1e-9)
            _, normal = elem.compute_sag_and_normal_at(x, y)
            assert np.allclose(np.asarray(normal), [0.0, 0.0, 1.0], atol=1e-9)

    def test_standalone_intersection_on_surface(self):
        def fn(x, y):
            return 0.05 * (x**2 + y**2)
        surf, _, _ = _freeform(fn, n=41)
        origin = jnp.array([0.2, -0.1, 5.0])
        direction = jnp.array([0.01, -0.02, -1.0])
        direction = direction / jnp.linalg.norm(direction)
        t, point, _ = surf.intersect_at(0, origin, direction)
        hit = origin + t * direction
        assert np.allclose(np.asarray(point), np.asarray(hit), atol=1e-6)
        z_surf = surf.sag_at(0, float(hit[0]), float(hit[1]))
        assert float(hit[2]) == pytest.approx(float(z_surf), abs=1e-6)

    def test_decenter_shifts_sampling(self):
        def fn(x, y):
            return 0.02 * x**2 + 0.01 * y**2
        offset = (0.1, -0.15)
        surf, _, _ = _freeform(fn, offset=offset)
        centered, _, _ = _freeform(fn, offset=(0.0, 0.0))
        assert float(surf.sag_at(0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
        x, y = 0.1, 0.05
        ox, oy = offset
        expected = (
            float(centered._index(0)._sag_intrinsic(x + ox, y + oy))
            - float(centered._index(0)._sag_intrinsic(ox, oy))
        )
        assert float(surf.sag_at(0, x, y)) == pytest.approx(expected, abs=1e-9)

    def test_index_slicing(self):
        g0 = jnp.zeros((5, 5))
        g1 = jnp.ones((5, 5))
        surf = FreeformSurfaceGroup.from_extent(
            jnp.stack([g0, g1]), jnp.array([0.5, 0.3]), 0.5
        )
        elem = surf._index(1)
        assert elem.grid_z.shape == (5, 5)
        assert np.allclose(np.asarray(elem.grid_z), 1.0)
        assert float(elem.x0) == pytest.approx(-0.3)

    def test_validation(self):
        with pytest.raises(ValueError):
            FreeformSurfaceGroup.from_extent(jnp.zeros((5, 5)), 0.5, 0.5)  # not 3D
        with pytest.raises(ValueError):
            FreeformSurfaceGroup(  # 1xW grid too small
                grid_z=jnp.zeros((1, 1, 4)), x0=0.0, y0=0.0, dx=1.0, dy=1.0,
            )


class TestFreeformInSum:
    def test_sum_is_additive(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.05]), conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)), offsets=jnp.zeros((1, 2)),
        )
        free, _, _ = _freeform(lambda x, y: 0.002 * jnp.sin(5 * x), n=41)
        s = SumSurfaceGroup([asph, free])
        for x, y in [(0.2, 0.1), (-0.3, 0.25)]:
            expected = float(asph.sag_at(0, x, y)) + float(free.sag_at(0, x, y))
            assert float(s.sag_at(0, x, y)) == pytest.approx(expected, abs=1e-9)

    def test_sum_intersection_and_normal(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.06]), conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)), offsets=jnp.zeros((1, 2)),
        )
        free, _, _ = _freeform(lambda x, y: 0.001 * (x**2 - y**2), n=41)
        s = SumSurfaceGroup([asph, free])  # asphere first -> conic initial guess
        origin = jnp.array([0.25, -0.15, 8.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        t, point, normal = s.intersect_at(0, origin, direction)
        z_surf = s.sag_at(0, float(point[0]), float(point[1]))
        assert float(point[2]) == pytest.approx(float(z_surf), abs=1e-6)
        # normal vs finite difference of the composite sag
        elem = s._index(0)
        dzdx, dzdy = _fd_slope(lambda xx, yy: elem._sag_local(xx, yy),
                               float(point[0]), float(point[1]))
        expected = np.array([-dzdx, -dzdy, 1.0])
        expected /= np.linalg.norm(expected)
        assert np.allclose(np.asarray(normal), expected, atol=1e-4)


def _mirror_group_with_surface(surface, radius=0.4):
    n = surface.offsets.shape[0]
    aperture = DiskAperture(radii=jnp.full(n, radius), inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=0,
        n_samples=64,
    )


class TestFreeformSerializationGuard:
    """Freeform grids are not YAML-serializable; saving must fail loudly."""

    def test_save_raises(self):
        from iactrace import Telescope
        from iactrace.io import telescope_to_dict

        free, _, _ = _freeform(lambda x, y: 0.002 * (x**2 - y**2), n=9)
        tel = Telescope(mirror_groups=[_mirror_group_with_surface(free)], name="f")
        with pytest.raises(ValueError, match="FreeformSurfaceGroup"):
            telescope_to_dict(tel)


class TestFreeformRenderPipeline:
    def test_transform_to_world(self):
        free, _, _ = _freeform(lambda x, y: 0.003 * (x**2 - y**2), n=31, half=0.5)
        group = _mirror_group_with_surface(free)
        points, normals, _ = group.transform_to_world()
        assert points.shape == (1, 64, 3)
        assert jnp.all(jnp.isfinite(points))
        assert jnp.allclose(jnp.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)

    def test_intersect_at_vmapped(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.06]), conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)), offsets=jnp.zeros((1, 2)),
        )
        free, _, _ = _freeform(lambda x, y: 0.002 * jnp.sin(6 * x), n=41, half=0.5)
        group = _mirror_group_with_surface(SumSurfaceGroup([asph, free]))
        n_rays = 32
        key = jax.random.key(2)
        xy = jax.random.uniform(key, (n_rays, 2), minval=-0.3, maxval=0.3)
        origins = jnp.concatenate([xy, jnp.full((n_rays, 1), 6.0)], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        ts, pts, _ = jax.vmap(
            lambda o, d: group.surface.intersect_at(0, o, d)
        )(origins, directions)
        assert jnp.all(jnp.isfinite(ts))
        z_surf = jax.vmap(lambda p: group.surface.sag_at(0, p[0], p[1]))(pts)
        assert jnp.allclose(pts[:, 2], z_surf, atol=1e-6)
