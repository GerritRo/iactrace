import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from iactrace import (
    ConstantIndex,
    ConstantSpectrum,
    SellmeierIndex,
    TabulatedIndex,
    TabulatedQE,
    TabulatedResponse,
    TabulatedSpectrum,
)
from iactrace.core.interactions import RefractInteraction, SlabInteraction
from iactrace.core.ray_bundle import DEFAULT_WAVELENGTH, RayBundle
from iactrace.core.responses import ConstantResponse
from iactrace.core.spectrum import as_spectrum

from ._helpers import make_simple_telescope


def _bundle(n=3, wavelength=None):
    return RayBundle(
        origins=jnp.zeros((n, 3)),
        directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1)),
        values=jnp.ones(n),
        path_length=jnp.zeros(n),
        n=jnp.ones(n),
        wavelength=wavelength,
    )


class TestRayBundleField:
    def test_default_is_reference_wavelength(self):
        rb = _bundle()
        assert rb.wavelength.shape == (3,)
        # Every ray always has a real wavelength -- no sentinel / NaN.
        assert bool(jnp.all(jnp.isfinite(rb.wavelength)))
        assert jnp.allclose(rb.wavelength, DEFAULT_WAVELENGTH)

    def test_explicit_value_stored(self):
        rb = _bundle(wavelength=jnp.full(3, 500.0))
        assert jnp.allclose(rb.wavelength, 500.0)

    def test_replace_and_to_frame_preserve_wavelength(self):
        rb = _bundle(wavelength=jnp.array([300.0, 400.0, 500.0]))
        assert jnp.allclose(rb.replace(values=jnp.zeros(3)).wavelength, rb.wavelength)
        moved = rb.to_frame(jnp.array([1.0, 2.0, 3.0]), jnp.zeros(3))
        assert jnp.allclose(moved.wavelength, rb.wavelength)


class TestConstantResponse:
    def test_flat_in_both_axes(self):
        c = ConstantResponse(values=jnp.array([0.9, 0.8]))
        cos, idx = jnp.array([1.0, 0.5]), jnp.array([0, 1])
        assert jnp.allclose(c(cos, idx), c(cos, idx, jnp.array([300.0, 600.0])))


class TestTabulatedResponse:
    def test_angle_only_is_degenerate_single_wavelength(self):
        c = TabulatedResponse.from_degrees(angles_deg=[0.0, 90.0], values=[1.0, 0.0], n_elements=1)
        assert c.values.shape == (1, 2, 1)  # (N, Kc, Kw=1)
        idx = jnp.array([0])
        # Wavelength is irrelevant for an angle-only curve.
        assert jnp.allclose(c(jnp.array([0.5]), idx), 0.5, atol=1e-9)
        assert jnp.allclose(c(jnp.array([0.5]), idx, jnp.array([555.0])), 0.5, atol=1e-9)

    def test_from_degrees_2d_shapes_and_broadcast(self):
        c = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            wavelengths=[300.0, 600.0],
            values=[[0.5, 1.0], [0.5, 1.0]],  # (Kc, Kw) broadcast to N
            n_elements=3,
        )
        assert c.values.shape == (3, 2, 2)

    def test_rejects_wrong_n_elements(self):
        with pytest.raises(ValueError, match="must match"):
            TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0],
                wavelengths=[300.0, 600.0],
                values=jnp.zeros((2, 2, 2)),  # N=2
                n_elements=3,
            )

    def test_rejects_axis_length_mismatch(self):
        # angle-only: 3 angles but 2 values
        with pytest.raises(ValueError, match="angle axis"):
            TabulatedResponse.from_degrees(
                angles_deg=[0.0, 45.0, 90.0], values=[1.0, 0.0], n_elements=1
            )
        # 2D grid: cos axis (3) doesn't match angles (2)
        with pytest.raises(ValueError, match="must match"):
            TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0],
                wavelengths=[300.0, 600.0],
                values=jnp.zeros((1, 3, 2)),
                n_elements=1,
            )

    def test_none_wavelength_falls_back_to_first_sample(self):
        # Wavelength-varying, angle-flat: 0.5 at 300 nm, 1.0 at 600 nm.
        c = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            wavelengths=[300.0, 600.0],
            values=[[0.5, 1.0], [0.5, 1.0]],
            n_elements=1,
        )
        # No wavelength given -> uses the first tabulated wavelength (300 nm).
        assert jnp.allclose(c(jnp.array([1.0]), jnp.array([0])), 0.5, atol=1e-9)

    def test_exact_at_grid_knots(self):
        c = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            wavelengths=[300.0, 600.0],
            values=[[0.5, 1.0], [0.5, 1.0]],
            n_elements=1,
        )
        idx = jnp.array([0])
        assert jnp.allclose(c(jnp.array([1.0]), idx, jnp.array([300.0])), 0.5, atol=1e-9)
        assert jnp.allclose(c(jnp.array([0.3]), idx, jnp.array([600.0])), 1.0, atol=1e-9)
        assert jnp.allclose(c(jnp.array([0.8]), idx, jnp.array([450.0])), 0.75, atol=1e-9)

    def test_bilinear_between_knots(self):
        c = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            wavelengths=[300.0, 600.0],
            values=[[0.0, 0.5], [0.5, 1.0]],
            n_elements=1,
        )
        got = c(jnp.array([0.5]), jnp.array([0]), jnp.array([450.0]))
        assert jnp.allclose(got, 0.5, atol=1e-9)


class TestRefractiveIndex:
    def test_constant_ignores_wavelength(self):
        idx = ConstantIndex(jnp.array([1.5, 1.6]))
        got = idx.n_at(jnp.array([0, 1, 0]), jnp.array([300.0, 600.0, 900.0]))
        assert jnp.allclose(got, jnp.array([1.5, 1.6, 1.5]))
        assert jnp.allclose(idx.reference(), jnp.array([1.5, 1.6]))

    def test_tabulated_interpolates(self):
        d = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=1)
        got = d.n_at(jnp.array([0, 0, 0]), jnp.array([300.0, 450.0, 600.0]))
        assert jnp.allclose(got, jnp.array([1.4, 1.5, 1.6]), atol=1e-9)
        # reference() evaluates at DEFAULT_WAVELENGTH.
        expected_ref = jnp.interp(
            DEFAULT_WAVELENGTH, jnp.array([300.0, 600.0]), jnp.array([1.4, 1.6])
        )
        assert jnp.allclose(d.reference(), expected_ref)

    def test_tabulated_from_table_broadcast_and_reject(self):
        d = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=3)
        assert d.n_values.shape == (3, 2)
        with pytest.raises(ValueError, match="must match"):
            TabulatedIndex.from_table([300.0, 600.0], jnp.array([[1.4, 1.6]]), n_elements=3)

    def test_from_table_rejects_wavelength_axis_mismatch(self):
        with pytest.raises(ValueError, match="wavelength axis"):
            TabulatedIndex.from_table([300.0, 600.0, 900.0], [[1.4, 1.6]], n_elements=1)

    def test_reference_uses_design_wavelength(self):
        d = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=1)
        assert jnp.allclose(d.reference(300.0), 1.4)
        assert jnp.allclose(d.reference(600.0), 1.6)
        # A constant index ignores the design wavelength.
        assert jnp.allclose(ConstantIndex(jnp.array([1.5])).reference(999.0), 1.5)

    def test_n_elements(self):
        assert ConstantIndex(jnp.array([1.5, 1.6])).n_elements == 2
        assert TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=3).n_elements == 3
        assert SellmeierIndex(b=jnp.ones((4, 1)), c=jnp.full((4, 1), 2.0)).n_elements == 4

    def test_sellmeier_matches_formula_and_selects_element(self):
        b = jnp.array([[1.0], [1.2]])
        c = jnp.array([[1.0], [2.0]])
        d = SellmeierIndex(b=b, c=c)
        wl = jnp.array([2.0, 3.0])
        lam2 = wl**2
        expected = jnp.sqrt(1.0 + jnp.array([1.0, 1.2]) * lam2 / (lam2 - jnp.array([1.0, 2.0])))
        assert jnp.allclose(d.n_at(jnp.array([0, 1]), wl), expected, atol=1e-9)


class TestDispersiveRefraction:
    """A non-constant index makes refraction / OPL wavelength-dependent."""

    def _tilted_ray(self):
        theta = jnp.deg2rad(30.0)
        directions = jnp.array([[jnp.sin(theta), 0.0, -jnp.cos(theta)]])
        normals = jnp.array([[0.0, 0.0, 1.0]])
        return directions, normals

    def test_refract_bends_differently_per_wavelength(self):
        idx = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=1)
        ri = RefractInteraction(index=idx, transmittance_curve=None, transmittance=jnp.ones(1))
        directions, normals = self._tilted_ray()
        pts, elem = jnp.zeros((1, 3)), jnp.array([0])
        d_blue, _, _, _, n_blue = ri.apply(
            directions, normals, pts, elem, jnp.ones(1), jnp.array([300.0])
        )
        d_red, _, _, _, n_red = ri.apply(
            directions, normals, pts, elem, jnp.ones(1), jnp.array([600.0])
        )
        assert jnp.allclose(n_blue, 1.4) and jnp.allclose(n_red, 1.6)
        assert not jnp.allclose(d_blue, d_red, atol=1e-6)

    def test_constant_index_is_wavelength_independent(self):
        ri = RefractInteraction(
            index=ConstantIndex(jnp.array([1.5])),
            transmittance_curve=None,
            transmittance=jnp.ones(1),
        )
        directions, normals = self._tilted_ray()
        _, _, _, _, n_out = ri.apply(
            directions, normals, jnp.zeros((1, 3)), jnp.array([0]), jnp.ones(1), jnp.array([300.0])
        )
        assert jnp.allclose(n_out, 1.5)

    def test_slab_dispersion_changes_opl(self):
        idx = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=1)
        slab = SlabInteraction(
            index=idx,
            thickness=jnp.array([0.01]),
            transmittance_curve=None,
            transmittance=jnp.ones(1),
        )
        directions, normals = self._tilted_ray()
        pts, elem = jnp.zeros((1, 3)), jnp.array([0])
        *_, opl_blue, n_blue = slab.apply(
            directions, normals, pts, elem, jnp.ones(1), jnp.array([300.0])
        )
        *_, opl_red, n_red = slab.apply(
            directions, normals, pts, elem, jnp.ones(1), jnp.array([600.0])
        )
        assert jnp.allclose(n_blue, 1.0) and jnp.allclose(n_red, 1.0)  # ambient unchanged
        assert bool(jnp.all(opl_red > opl_blue))


class TestFocalDesignWavelength:
    """focal_scale evaluates a dispersive lens index at an explicit design wavelength."""

    def test_dispersive_focal_scale_tracks_design_wavelength(self):
        idx = TabulatedIndex.from_table([300.0, 600.0], [1.4, 1.6], n_elements=1)
        ri = RefractInteraction(index=idx, transmittance_curve=None, transmittance=jnp.ones(1))
        assert jnp.allclose(ri.focal_scale(1.0, 300.0), 0.4)  # (1.4 - 1.0)
        assert jnp.allclose(ri.focal_scale(1.0, 600.0), 0.6)  # (1.6 - 1.0)

    def test_constant_focal_scale_ignores_design_wavelength(self):
        ri = RefractInteraction(
            index=ConstantIndex(jnp.array([1.5])),
            transmittance_curve=None,
            transmittance=jnp.ones(1),
        )
        assert jnp.allclose(ri.focal_scale(1.0, 300.0), ri.focal_scale(1.0, 600.0))
        assert jnp.allclose(ri.focal_scale(1.0), 0.5)  # default wavelength, (1.5 - 1.0)


class TestTabulatedQE:
    def test_interpolates_qe_at_ray_wavelength(self):
        qe = TabulatedQE.from_table([300.0, 600.0], [0.2, 0.8])
        rb = _bundle(n=3, wavelength=jnp.array([300.0, 450.0, 600.0]))
        out = qe.detect(rb)
        assert jnp.allclose(out.values, jnp.array([0.2, 0.5, 0.8]), atol=1e-9)

    def test_angle_dependent_qe_curve_uses_the_incidence_angle(self):
        # QE is an ordinary ResponseCurve now, so a photodetector honours its
        # angle axis the same way a mirror or a cone wall does.
        import numpy as np

        curve = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0], values=[0.9, 0.1], n_elements=1
        )
        det = TabulatedQE(qe_curve=curve)
        n = 2
        tilt = jnp.sqrt(0.5)
        rb = RayBundle(
            origins=jnp.zeros((n, 3)),
            directions=jnp.stack([jnp.array([0.0, 0.0, -1.0]), jnp.array([tilt, 0.0, -tilt])]),
            values=jnp.ones(n),
            path_length=jnp.zeros(n),
            n=jnp.ones(n),
            wavelength=jnp.full(n, 400.0),
        )
        out = np.asarray(det.detect(rb).values)
        np.testing.assert_allclose(out[0], 0.9, atol=1e-6)  # normal incidence
        assert out[1] < out[0]  # 45 deg sees less
        np.testing.assert_allclose(out[1], 0.1 + 0.8 * tilt, atol=1e-5)

    def test_bulk_qe_multiplies_the_curve(self):
        import numpy as np

        det = TabulatedQE.from_table([300.0, 600.0], [0.2, 0.8], bulk_qe=0.5)
        rb = _bundle(n=2, wavelength=jnp.array([300.0, 600.0]))
        np.testing.assert_allclose(np.asarray(det.detect(rb).values), [0.1, 0.4], atol=1e-6)

    def test_from_table_sorts(self):
        # The QE curve is an ordinary ResponseCurve, so the samples land on its
        # wavelength axis in ascending order however they were given.
        qe = TabulatedQE.from_table([600.0, 300.0], [0.8, 0.2])
        assert jnp.allclose(qe.qe_curve.wl_table, jnp.array([300.0, 600.0]))
        assert jnp.allclose(qe.qe_curve.values[0, 0], jnp.array([0.2, 0.8]))


def _mirror_with_wavelength_reflectivity(telescope, r_blue=0.5, r_red=1.0):
    """Attach an angle-flat, wavelength-ramped reflectivity to the primary."""
    coating = TabulatedResponse.from_degrees(
        angles_deg=[0.0, 90.0],
        wavelengths=[300.0, 600.0],
        values=[[r_blue, r_red], [r_blue, r_red]],
        n_elements=1,
    )
    return eqx.tree_at(
        lambda t: t.mirror_groups[0].interaction_module.reflectivity_curve,
        telescope,
        coating,
        is_leaf=lambda x: x is None,
    )


class TestRenderThreading:
    def test_materialised_bundle_carries_wavelength(self):
        telescope, _ = make_simple_telescope(n_samples=64)
        directions = jnp.array([[0.0, 0.0, -1.0]])
        rb = telescope.render(
            directions, jnp.array([1.0]), source_type="parallel", wavelength=532.0
        )
        mat = rb.materialise()
        live = mat.wavelength[mat.alive]
        assert live.size > 0
        assert jnp.allclose(live, 532.0)

    def test_default_render_is_monochromatic_at_reference(self):
        telescope, _ = make_simple_telescope(n_samples=64)
        rb = telescope.render(
            jnp.array([[0.0, 0.0, -1.0]]), jnp.array([1.0]), source_type="parallel"
        )
        mat = rb.materialise()
        assert jnp.allclose(mat.wavelength[mat.alive], DEFAULT_WAVELENGTH)

    def test_constant_mode_unaffected_by_wavelength(self):
        """A wavelength-independent mirror gives the same image regardless of the
        wavelength argument -- the threading never perturbs the result."""
        telescope, camera = make_simple_telescope(n_samples=128)
        directions = jnp.array([[0.0, 0.0, -1.0]])
        val = jnp.array([1.0])
        img_default = camera.image(telescope.render(directions, val, source_type="parallel"))
        img_500 = camera.image(
            telescope.render(directions, val, source_type="parallel", wavelength=500.0)
        )
        assert jnp.allclose(img_default, img_500)
        assert jnp.all(jnp.isfinite(img_default))


class TestSpectrum:
    def test_constant_spectrum(self):
        cs = ConstantSpectrum(jnp.asarray(500.0))
        assert jnp.allclose(cs.sample(jax.random.key(0), (5,)), 500.0)
        wl, w = cs.bins()
        assert wl.shape == (1,) and jnp.allclose(wl, 500.0) and jnp.allclose(w, 1.0)

    def test_as_spectrum_coerces(self):
        assert isinstance(as_spectrum(550.0), ConstantSpectrum)
        cs = ConstantSpectrum(jnp.asarray(1.0))
        assert as_spectrum(cs) is cs

    def test_tabulated_sampling_follows_density(self):
        wl = jnp.linspace(300.0, 600.0, 50)
        ts = TabulatedSpectrum.from_density(wl, 1.0 / wl**2)  # Cherenkov: short-wl heavy
        draws = ts.sample(jax.random.key(1), (100_000,))
        assert float(draws.min()) >= 300.0 and float(draws.max()) <= 600.0
        assert float((draws < 400).mean()) > float((draws > 500).mean())

    def test_sampled_density_matches_the_linear_ramp(self):
        """The density between two nodes must ramp, not come out flat.

        Interpolating the inverse CDF linearly gets every *segment mass* right
        while flattening the ramp inside it, so a many-node grid hides the
        error entirely. One segment with a steep ramp is what exposes it.
        """
        import numpy as np

        wl = jnp.array([300.0, 600.0])
        ts = TabulatedSpectrum(wl, jnp.array([1.0, 2.0]))
        draws = np.asarray(ts.sample(jax.random.key(3), (400_000,)))

        edges = np.linspace(300.0, 600.0, 7)
        hist, _ = np.histogram(draws, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        expected = np.interp(centres, [300.0, 600.0], [1.0, 2.0])
        np.testing.assert_allclose(hist / hist.mean(), expected / expected.mean(), rtol=0.02)

    def test_sampling_is_accurate_where_bins_quadrature_is_not(self):
        """Pins which estimator to trust when the two disagree.

        ``bins()`` is a 2-point-per-segment quadrature, exact only when the
        weighted integrand is linear on each segment. With a varying density
        against a convex curve it is not, and the quadrature carries a real
        error; the sampler reproduces the density, so it converges to the truth.
        """
        import numpy as np

        wl = jnp.array([300.0, 450.0, 600.0])
        density = jnp.array([1.0, 2.0, 1.0])
        curve = np.array([0.0, 0.1, 1.0])  # convex: mean of f != f of the mean
        ts = TabulatedSpectrum(wl, density)

        grid = np.linspace(300.0, 600.0, 200_001)
        p = np.interp(grid, np.asarray(wl), np.asarray(density))
        f = np.interp(grid, np.asarray(wl), curve)
        truth = np.trapezoid(p * f, grid) / np.trapezoid(p, grid)

        draws = np.asarray(ts.sample(jax.random.key(4), (400_000,)))
        sampled = float(np.interp(draws, np.asarray(wl), curve).mean())

        nodes, weights = ts.bins()
        quadrature = float(
            np.sum(np.asarray(weights) * np.interp(np.asarray(nodes), np.asarray(wl), curve))
        )

        assert sampled == pytest.approx(truth, rel=0.01)
        assert abs(quadrature - truth) > 0.05 * truth  # ...and quadrature is not

    def test_sample_handles_a_density_starting_at_zero(self):
        # p0 == 0 drives the quadratic's discriminant to zero at the segment
        # start, which is where a naive sqrt would hand back a NaN gradient.
        ts = TabulatedSpectrum(jnp.array([300.0, 600.0]), jnp.array([0.0, 1.0]))
        draws = ts.sample(jax.random.key(5), (10_000,))
        assert bool(jnp.all(jnp.isfinite(draws)))
        assert float(draws.min()) >= 300.0 and float(draws.max()) <= 600.0
        # ...and the draw favours the heavy end.
        assert float((draws > 500.0).mean()) > float((draws < 400.0).mean())

        grad = jax.grad(
            lambda d: (
                TabulatedSpectrum(jnp.array([300.0, 600.0]), d)
                .sample(jax.random.key(5), (1_000,))
                .mean()
            )
        )(jnp.array([0.0, 1.0]))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_sample_is_differentiable_in_the_density(self):
        # The draw is reparameterised, so the spectrum's parameters stay
        # reachable by grad -- that is what keeps a broadband render fittable.
        wl = jnp.array([300.0, 450.0, 600.0])

        def mean_wavelength(density):
            return TabulatedSpectrum(wl, density).sample(jax.random.key(6), (4_000,)).mean()

        grad = jax.grad(mean_wavelength)(jnp.array([1.0, 2.0, 1.0]))
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.abs(grad).sum()) > 0.0

    def test_tabulated_bins_normalized_and_sorted(self):
        ts = TabulatedSpectrum.from_density([600.0, 300.0], [2.0, 1.0])  # unsorted input
        assert jnp.allclose(ts.wavelengths, jnp.array([300.0, 600.0]))
        wl, w = ts.bins()
        assert jnp.allclose(w.sum(), 1.0)


class TestWavelengthSampling:
    """A render draws one wavelength per ray; it never replicates rays."""

    def _setup(self, n_samples=128):
        telescope, camera = make_simple_telescope(n_samples=n_samples)
        telescope = _mirror_with_wavelength_reflectivity(telescope, r_blue=0.5, r_red=1.0)
        return telescope, camera, jnp.array([[0.0, 0.0, -1.0]]), jnp.array([1.0])

    def test_constant_spectrum_equals_scalar(self):
        telescope, camera, directions, val = self._setup()
        a = camera.image(telescope.render(directions, val, "parallel", wavelength=550.0))
        b = camera.image(
            telescope.render(
                directions, val, "parallel", wavelength=ConstantSpectrum(jnp.asarray(550.0))
            )
        )
        assert jnp.allclose(a, b)

    def test_a_spectrum_costs_no_extra_rays(self):
        """The whole point of sampling over sweeping."""
        telescope, _, directions, val = self._setup(n_samples=32)
        mono = telescope.render(directions, val, "parallel", wavelength=450.0).materialise()
        flat = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        broad = telescope.render(directions, val, "parallel", wavelength=flat).materialise()
        assert broad.values.shape == mono.values.shape

    def test_an_array_is_rejected_with_a_pointer_to_the_manual_sweep(self):
        telescope, _, directions, val = self._setup(n_samples=16)
        with pytest.raises(ValueError, match="scalar wavelength or a Spectrum"):
            telescope.render(directions, val, "parallel", wavelength=[400.0, 500.0])

    def test_wavelengths_spread_across_the_band(self):
        import numpy as np

        telescope, _, directions, val = self._setup(n_samples=256)
        flat = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        wl = np.asarray(
            telescope.render(directions, val, "parallel", wavelength=flat).materialise().wavelength
        )
        assert wl.min() >= 300.0 - 1e-6 and wl.max() <= 600.0 + 1e-6
        assert wl.std() > 50.0  # genuinely drawn, not one shared value
        assert abs(wl.mean() - 450.0) < 25.0  # flat band -> centred

    def test_the_draw_is_reproducible(self):
        # The key is derived from the group's sample_key, so the same telescope
        # renders the same wavelengths every time.
        import numpy as np

        telescope, _, directions, val = self._setup(n_samples=32)
        flat = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        a = telescope.render(directions, val, "parallel", wavelength=flat).materialise()
        b = telescope.render(directions, val, "parallel", wavelength=flat).materialise()
        np.testing.assert_array_equal(np.asarray(a.wavelength), np.asarray(b.wavelength))

    def test_each_ray_sees_its_own_wavelength_reflectivity(self):
        """Exact check: geometry is wavelength-independent, so ray i is ray i.

        The mirror ramps 0.5 at 300 nm to 1.0 at 600 nm, so dividing the
        broadband values by a reference render at 600 nm (where R = 1) must
        recover each ray's own R(lambda).
        """
        import numpy as np

        telescope, _, directions, val = self._setup(n_samples=64)
        flat = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        broad = telescope.render(directions, val, "parallel", wavelength=flat).materialise()
        ref = telescope.render(directions, val, "parallel", wavelength=600.0).materialise()

        live = np.asarray(ref.values) > 0
        ratio = np.asarray(broad.values)[live] / np.asarray(ref.values)[live]
        expected = 0.5 + 0.5 * (np.asarray(broad.wavelength)[live] - 300.0) / 300.0
        np.testing.assert_allclose(ratio, expected, rtol=1e-5)

    def test_flux_is_preserved_across_the_band(self):
        telescope, _, directions, val = self._setup(n_samples=512)
        # A flat band with a flat mirror response: the broadband total must
        # match a monochromatic render, since a Spectrum is a distribution and
        # not a multiplier.
        telescope_plain, _ = make_simple_telescope(n_samples=512)
        mono = telescope_plain.render(directions, val, "parallel", wavelength=450.0).materialise()
        flat = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        broad = telescope_plain.render(directions, val, "parallel", wavelength=flat).materialise()
        assert float(broad.values.sum()) == pytest.approx(float(mono.values.sum()), rel=1e-5)

    def test_manual_sweep_reproduces_the_sampled_broadband_image(self):
        """The documented user-side sweep: loop over bins() yourself."""
        telescope, camera, directions, val = self._setup(n_samples=4096)
        spectrum = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])

        sampled = camera.image(telescope.render(directions, val, "parallel", wavelength=spectrum))
        wavelengths, weights = spectrum.bins()
        swept = sum(
            float(w)
            * camera.image(telescope.render(directions, val, "parallel", wavelength=float(wl)))
            for wl, w in zip(wavelengths, weights, strict=True)
        )
        # Monte Carlo vs quadrature: totals agree, within sampling noise.
        assert float(sampled.sum()) == pytest.approx(float(swept.sum()), rel=0.02)

    def test_image_and_response_matrix_accept_a_spectrum(self):
        telescope, camera, directions, val = self._setup(n_samples=64)
        spectrum = TabulatedSpectrum.from_density([350.0, 550.0], [1.0, 1.0])
        img = camera.image(telescope.render(directions, val, "parallel", wavelength=spectrum))
        assert float(img.sum()) > 0.0

        two = jnp.array([[0.0, 0.0, -1.0], [0.01, 0.0, -1.0]])
        rb = telescope.render(two, jnp.ones(2), "parallel", wavelength=spectrum)
        matrix = camera.response_matrix(rb)
        assert matrix.shape[0] == 2
        assert jnp.allclose(matrix.sum(axis=0), camera.image(rb), atol=1e-6)

    def test_trace_keeps_arrays_per_ray(self):
        # `trace` is handed its rays, so an array stays per-ray there; a
        # mismatched length is an error.
        telescope, _, _, _ = self._setup(n_samples=16)
        n = 6
        o = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.full(n, 10.0)], axis=1)
        d = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        wl = jnp.linspace(300.0, 600.0, n)
        rays = telescope.trace(o, d, jnp.ones(n), wavelength=wl).rays
        assert rays.wavelength.shape == (n,)
        assert jnp.allclose(rays.wavelength, wl)
        with pytest.raises(ValueError, match="per-ray"):
            telescope.trace(o, d, jnp.ones(n), wavelength=jnp.array([400.0, 500.0]))

    def test_trace_samples_a_spectrum_per_ray(self):
        telescope, _, _, _ = self._setup(n_samples=16)
        n = 512
        o = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.full(n, 10.0)], axis=1)
        d = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        spectrum = TabulatedSpectrum.from_density([300.0, 600.0], [1.0, 1.0])
        rays = telescope.trace(o, d, jnp.ones(n), wavelength=spectrum).rays
        assert rays.wavelength.shape == (n,)  # not K * n
        assert float(rays.wavelength.min()) >= 300.0 - 1e-6
        assert float(rays.wavelength.max()) <= 600.0 + 1e-6
        assert float(rays.wavelength.std()) > 50.0


class TestDifferentiability:
    def _setup(self):
        telescope, camera = make_simple_telescope(n_samples=64)
        telescope = _mirror_with_wavelength_reflectivity(telescope)
        return telescope, camera, jnp.array([[0.0, 0.0, -1.0]]), jnp.array([1.0])

    def test_grad_single_pass_through_spectrum_density(self):
        telescope, camera, directions, val = self._setup()

        def total(density):
            spectrum = TabulatedSpectrum(jnp.array([300.0, 600.0]), density)
            rays = telescope.render(directions, val, "parallel", wavelength=spectrum)
            return camera.image(rays).sum()

        g = jax.grad(total)(jnp.array([1.0, 2.0]))
        assert jnp.all(jnp.isfinite(g))


class TestCoatingFromWavelengths:
    def test_angle_flat_r_of_lambda(self):
        import numpy as np

        c = TabulatedResponse.from_wavelengths([300.0, 500.0], [0.5, 0.9], n_elements=1)
        wl = jnp.array([300.0, 400.0, 500.0])
        idx = jnp.zeros(3, dtype=jnp.int32)
        r_normal = np.asarray(c(jnp.ones(3), idx, wl))
        r_grazing = np.asarray(c(jnp.full(3, 0.1), idx, wl))
        np.testing.assert_allclose(r_normal, [0.5, 0.7, 0.9], atol=1e-5)
        np.testing.assert_allclose(r_normal, r_grazing, atol=1e-6)  # flat in angle

    def test_matches_duplicated_angle_grid(self):
        import numpy as np

        c1 = TabulatedResponse.from_wavelengths([300.0, 500.0], [0.5, 0.9], n_elements=1)
        c2 = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            values=[[0.5, 0.9], [0.5, 0.9]],
            n_elements=1,
            wavelengths=[300.0, 500.0],
        )
        wl = jnp.array([320.0, 450.0])
        idx = jnp.zeros(2, dtype=jnp.int32)
        np.testing.assert_allclose(
            np.asarray(c1(jnp.ones(2), idx, wl)),
            np.asarray(c2(jnp.ones(2), idx, wl)),
            atol=1e-6,
        )

    def test_per_element_rows(self):
        import numpy as np

        c = TabulatedResponse.from_wavelengths(
            [300.0, 500.0], [[0.5, 0.9], [0.2, 0.4]], n_elements=2
        )
        wl = jnp.array([300.0, 300.0])
        r = np.asarray(c(jnp.ones(2), jnp.array([0, 1], dtype=jnp.int32), wl))
        np.testing.assert_allclose(r, [0.5, 0.2], atol=1e-5)


class TestFocalSurfaceWavelength:
    def test_hits_carry_ray_wavelength(self):
        import numpy as np

        from iactrace.analysis import FlatFocalPlane

        telescope, _ = make_simple_telescope(n_samples=64)
        n = 32
        o = jnp.stack([jnp.linspace(-0.3, 0.3, n), jnp.zeros(n), jnp.full(n, 10.0)], axis=1)
        d = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        rays = telescope.trace(o, d, jnp.ones(n), wavelength=477.0).rays
        hits = FlatFocalPlane(
            position=jnp.array([0.0, 0.0, -5.0]), rotation=jnp.zeros(3)
        ).intersect(rays)
        assert hits.wavelength.shape == (n,)
        np.testing.assert_allclose(np.asarray(hits.wavelength), 477.0)

    def test_polychromatic_hits_preserve_per_ray_wavelength(self):
        import numpy as np

        from iactrace.analysis import FlatFocalPlane

        telescope, _ = make_simple_telescope(n_samples=16)
        n = 6
        o = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.full(n, 10.0)], axis=1)
        d = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        wl = jnp.linspace(300.0, 600.0, n)
        rays = telescope.trace(o, d, jnp.ones(n), wavelength=wl).rays
        hits = FlatFocalPlane(
            position=jnp.array([0.0, 0.0, -5.0]), rotation=jnp.zeros(3)
        ).intersect(rays)
        np.testing.assert_allclose(np.asarray(hits.wavelength), np.asarray(wl))


class TestPMTWavelength:
    def _rays(self, wl):
        n = wl.shape[0]
        return RayBundle(
            origins=jnp.zeros((n, 3)),
            directions=jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1)),
            values=jnp.ones(n),
            path_length=jnp.zeros(n),
            n=jnp.ones(n),
            wavelength=wl,
        )

    def test_qe_curve_varies_with_wavelength(self):
        import numpy as np

        from iactrace.camera.detector import PMT

        curve = TabulatedResponse.from_wavelengths(
            [300.0, 400.0, 500.0], [0.1, 0.4, 0.2], n_elements=1
        )
        pmt = PMT(qe=0.5, qe_curve=curve, face_radius=0.02)
        wl = jnp.array([300.0, 400.0, 500.0])
        out = np.asarray(pmt.detect(self._rays(wl)).values)
        # qe (bulk 0.5) * curve(lambda); no window -> exactly 0.5 * curve
        np.testing.assert_allclose(out, 0.5 * np.array([0.1, 0.4, 0.2]), atol=1e-6)

    def test_dispersive_window_changes_transmittance_with_wavelength(self):
        import numpy as np

        from iactrace.camera.detector import PMT

        bk7 = SellmeierIndex(
            b=jnp.array([[1.03961212, 0.231792344, 1.01046945]]),
            c=jnp.array([[6.00069867e3, 2.00179144e4, 1.03560653e8]]),
        )
        pmt = PMT(qe=1.0, window_index=bk7, face_radius=0.02)
        wl = jnp.array([350.0, 650.0])
        out = np.asarray(pmt.detect(self._rays(wl)).values)
        # Higher index (blue) -> lower Fresnel transmittance at normal incidence.
        assert out[0] < out[1]

    def test_one_window_parameter_takes_both_forms(self):
        # window_index is the single polymorphic index parameter: a number is
        # the non-dispersive case of the very same argument that takes a model.
        from iactrace.camera.detector import PMT
        from iactrace.core.refractive_index import ConstantIndex

        bk7 = SellmeierIndex(b=jnp.ones((1, 1)), c=jnp.full((1, 1), 1e7))
        assert isinstance(PMT(window_index=1.5, face_radius=0.02).window_index, ConstantIndex)
        assert PMT(window_index=bk7, face_radius=0.02).window_index is bk7
        with pytest.raises(ValueError, match="must be > 1.0"):
            PMT(window_index=0.9, face_radius=0.02)

    def test_scalar_pmt_unchanged(self):
        from iactrace.camera.detector import PMT

        pmt = PMT(qe=0.25, window_index=1.48, face_radius=0.02)
        assert pmt.qe_curve is None
        wl = jnp.array([400.0])
        idx = jnp.zeros(1, dtype=jnp.int32)
        assert float(pmt.window_index.n_at(idx, wl)[0]) == pytest.approx(1.48)
