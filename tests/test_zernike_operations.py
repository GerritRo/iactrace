"""Tests for Zernike figure-error operations and surface capability dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Telescope
from iactrace.core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from iactrace.telescope import operations as ops

from ._helpers import mirror_group_with_surface


@pytest.fixture
def asphere_telescope():
    n = 2
    surface = AsphericSurfaceGroup(
        curvatures=jnp.array([0.1, 0.2]),
        conics=jnp.array([-1.0, -1.0]),
        aspherics=jnp.zeros((n, 0)),
        offsets=jnp.zeros((n, 2)),
    )
    group = mirror_group_with_surface(surface, radius=jnp.array([0.5, 0.5]))
    return Telescope(mirror_groups=[group], name="t")


class TestApplyZernikeError:
    def test_wraps_bare_asphere_in_sum(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        # asphere component preserved unchanged
        asph = ops._asphere_of(surface)
        assert isinstance(asph, AsphericSurfaceGroup)
        assert np.allclose(np.asarray(asph.curvatures), [0.1, 0.2])
        # zernike component present
        zg = ops._zernike_of(surface)
        assert isinstance(zg, ZernikeSurfaceGroup)

    def test_coefficients_match_expected_draw(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        zg = ops._zernike_of(tel.stage(0).surface)
        expected = jax.random.normal(random_key, (2, sigmas.shape[0])) * sigmas[None, :]
        assert np.allclose(np.asarray(zg.coeffs), np.asarray(expected))

    def test_r_norm_from_aperture(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3])
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        zg = ops._zernike_of(tel.stage(0).surface)
        assert np.allclose(np.asarray(zg.r_norm), [0.5, 0.5])

    def test_accumulates(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4, 5e-4])
        k1, k2 = jax.random.split(random_key)
        tel1 = ops.apply_zernike_error(asphere_telescope, 0, sigmas, k1)
        tel2 = ops.apply_zernike_error(tel1, 0, sigmas, k2)
        c1 = ops._zernike_of(tel1.stage(0).surface).coeffs
        c2 = ops._zernike_of(tel2.stage(0).surface).coeffs
        draw2 = jax.random.normal(k2, (2, sigmas.shape[0])) * sigmas[None, :]
        # second application adds its draw on top of the first
        assert np.allclose(np.asarray(c2), np.asarray(c1 + draw2))
        # still a single Zernike term (not two)
        comps = tel2.stage(0).surface.components
        assert sum(isinstance(c, ZernikeSurfaceGroup) for c in comps) == 1

    def test_changes_surface_sag(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3])  # defocus
        tel = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        s0 = asphere_telescope.stage(0).surface
        s1 = tel.stage(0).surface
        z_before = float(s0.sag_at(0, 0.3, 0.2))
        z_after = float(s1.sag_at(0, 0.3, 0.2))
        assert not np.isclose(z_before, z_after)

    def test_too_many_modes_raises(self, asphere_telescope, random_key):
        with pytest.raises(ValueError):
            ops.apply_zernike_error(asphere_telescope, 0, jnp.zeros(12), random_key)

    def test_deterministic_and_key_dependent(self, asphere_telescope, random_key):
        sigmas = jnp.array([0.0, 0.0, 0.0, 1e-3, 5e-4])
        a = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        b = ops.apply_zernike_error(asphere_telescope, 0, sigmas, random_key)
        c = ops.apply_zernike_error(asphere_telescope, 0, sigmas, jax.random.key(7))
        ca = ops._zernike_of(a.stage(0).surface).coeffs
        cb = ops._zernike_of(b.stage(0).surface).coeffs
        cc = ops._zernike_of(c.stage(0).surface).coeffs
        assert np.allclose(np.asarray(ca), np.asarray(cb))
        assert not np.allclose(np.asarray(ca), np.asarray(cc))


class TestNamedAberrations:
    def _columns(self, tel):
        zg = ops._zernike_of(tel.stage(0).surface)
        return np.asarray(zg.coeffs)

    def test_astigmatism_only_z5_z6(self, asphere_telescope, random_key):
        tel = ops.apply_astigmatism(asphere_telescope, 0, 1e-3, random_key)
        cols = self._columns(tel)
        assert cols.shape[1] == 6
        # piston/tilt/defocus columns are zero; astig columns nonzero
        assert np.allclose(cols[:, :4], 0.0)
        assert np.any(cols[:, 4:6] != 0.0)

    def test_coma_only_z7_z8(self, asphere_telescope, random_key):
        tel = ops.apply_coma(asphere_telescope, 0, 1e-3, random_key)
        cols = self._columns(tel)
        assert cols.shape[1] == 8
        assert np.allclose(cols[:, :6], 0.0)
        assert np.any(cols[:, 6:8] != 0.0)

    def test_trefoil_only_z9_z10(self, asphere_telescope, random_key):
        tel = ops.apply_trefoil(asphere_telescope, 0, 1e-3, random_key)
        cols = self._columns(tel)
        assert cols.shape[1] == 10
        assert np.allclose(cols[:, :8], 0.0)
        assert np.any(cols[:, 8:10] != 0.0)

    def test_telescope_method(self, asphere_telescope, random_key):
        tel = asphere_telescope.apply_astigmatism(0, 1e-3, random_key)
        assert isinstance(tel.stage(0).surface, SumSurfaceGroup)


class TestCapabilityDispatchThroughSum:
    """Prescription operations keep working after a Zernike term is added."""

    def test_set_curvatures_through_sum(self, asphere_telescope, random_key):
        tel = ops.apply_zernike_error(
            asphere_telescope, 0, jnp.array([0.0, 0.0, 0.0, 1e-3]), random_key
        )
        tel = ops.set_curvatures(tel, 0, jnp.array([0.3, 0.4]))
        asph = ops._asphere_of(tel.stage(0).surface)
        assert np.allclose(np.asarray(asph.curvatures), [0.3, 0.4])
        # Zernike term untouched
        assert ops._zernike_of(tel.stage(0).surface) is not None

    def test_set_conics_through_sum(self, asphere_telescope, random_key):
        tel = asphere_telescope.apply_astigmatism(0, 1e-3, random_key)
        tel = ops.set_conics(tel, 0, jnp.array([0.0, 0.0]))
        asph = ops._asphere_of(tel.stage(0).surface)
        assert np.allclose(np.asarray(asph.conics), [0.0, 0.0])

    def test_focal_error_through_sum(self, asphere_telescope, random_key):
        k1, k2 = jax.random.split(random_key)
        tel = ops.apply_zernike_error(asphere_telescope, 0, jnp.array([0.0, 0.0, 0.0, 1e-3]), k1)
        c_before = ops._asphere_of(tel.stage(0).surface).curvatures
        tel = ops.apply_focal_error(tel, 0, 0.05, k2)
        c_after = ops._asphere_of(tel.stage(0).surface).curvatures
        assert not np.allclose(np.asarray(c_before), np.asarray(c_after))

    def test_no_asphere_raises(self, random_key):
        """A standalone-Zernike stage rejects aspheric prescription ops."""
        zg = ZernikeSurfaceGroup(coeffs=jnp.zeros((1, 4)), r_norm=jnp.ones(1))
        group = mirror_group_with_surface(zg, radius=jnp.array([0.5]))
        tel = Telescope(mirror_groups=[group], name="z")
        with pytest.raises(ValueError, match="no aspheric surface"):
            ops.set_curvatures(tel, 0, jnp.array([0.1]))


class TestEndToEndTrace:
    def test_zernike_error_perturbs_trace(self, asphere_telescope, random_key):
        n_rays = 64
        key = jax.random.key(3)
        xy = jax.random.uniform(key, (n_rays, 2), minval=-0.3, maxval=0.3)
        origins = jnp.concatenate([xy, jnp.full((n_rays, 1), 5.0)], axis=1)
        directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        values = jnp.ones(n_rays)

        rb_clean = asphere_telescope.trace(origins, directions, values)
        tel = asphere_telescope.apply_astigmatism(0, 5e-3, random_key)
        rb_pert = tel.trace(origins, directions, values)

        assert jnp.all(jnp.isfinite(rb_pert.directions))
        # the reflected directions should change under a real figure error
        assert not jnp.allclose(rb_clean.directions, rb_pert.directions)
