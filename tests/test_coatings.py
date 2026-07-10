"""Tests for :mod:`iactrace.core.coatings` and its integration with
:class:`ReflectInteraction`, :class:`RefractInteraction`, and
:class:`SlabInteraction`.
"""

import jax
import jax.numpy as jnp
import pytest

from iactrace.core.coatings import (
    ConstantCoating,
    TabulatedCoating,
    fresnel_unpolarized,
)
from iactrace.core.interactions import (
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
)


class TestConstantCoating:
    """Constant value."""

    def test_returns_per_element_value(self):
        coating = ConstantCoating(values=jnp.array([0.9, 0.85, 0.7]))
        idx = jnp.array([0, 1, 2, 1])
        cos = jnp.array([1.0, 0.5, 0.1, 0.9])
        out = coating(cos, idx)
        assert jnp.allclose(out, jnp.array([0.9, 0.85, 0.7, 0.85]))

    def test_ignores_angle(self):
        coating = ConstantCoating(values=jnp.array([0.9, 0.9, 0.9]))
        idx = jnp.zeros(5, dtype=jnp.int32)
        for c in (1.0, 0.5, 0.0):
            out = coating(jnp.full(5, c), idx)
            assert jnp.allclose(out, 0.9)


class TestTabulatedCoating:
    """Linear interpolation over a precomputed cos(angle) table."""

    def test_from_degrees_shared_curve(self):
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=[1.0, 0.7, 0.2],
            n_elements=3,
        )
        assert coating.cos_table.shape == (3,)
        # All three elements share the same curve.
        assert coating.values.shape == (3, 3)
        assert jnp.allclose(coating.values[0], coating.values[1])
        assert jnp.allclose(coating.values[0], coating.values[2])

    def test_from_degrees_per_element(self):
        per_elem = jnp.array([[1.0, 0.7, 0.2], [0.9, 0.5, 0.1], [0.95, 0.6, 0.15]])
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=per_elem,
            n_elements=3,
        )
        assert coating.values.shape == (3, 3)

    def test_from_degrees_rejects_wrong_n(self):
        with pytest.raises(ValueError, match="must match"):
            TabulatedCoating.from_degrees(
                angles_deg=[0.0, 90.0],
                values=jnp.array([[1.0, 0.2], [0.9, 0.1]]),
                n_elements=3,
            )

    def test_exact_at_knots(self):
        # Curve: R(0 deg) = 0.9, R(60 deg) = 0.4 (cos = 0.5).
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 60.0],
            values=[0.9, 0.4],
            n_elements=1,
        )
        # Normal incidence
        out = coating(jnp.array([1.0]), jnp.array([0]))
        assert jnp.allclose(out, 0.9, atol=1e-6)
        # 60 deg incidence (cos = 0.5)
        out = coating(jnp.array([0.5]), jnp.array([0]))
        assert jnp.allclose(out, 0.4, atol=1e-6)

    def test_linear_interpolation_at_midpoint(self):
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 90.0],  # cos 1.0 -> 0.0
            values=[1.0, 0.0],
            n_elements=1,
        )
        out = coating(jnp.array([0.5]), jnp.array([0]))  # cos = 0.5
        assert jnp.allclose(out, 0.5, atol=1e-6)

    def test_unsorted_input_normalized(self):
        c1 = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=[1.0, 0.7, 0.2],
            n_elements=1,
        )
        c2 = TabulatedCoating.from_degrees(
            angles_deg=[90.0, 0.0, 45.0],  # shuffled
            values=[0.2, 1.0, 0.7],
            n_elements=1,
        )
        cos = jnp.array([1.0, 0.5, 0.0])
        idx = jnp.array([0, 0, 0])
        assert jnp.allclose(c1(cos, idx), c2(cos, idx))


class TestDefaults:
    """``coating=None`` means: identity for mirrors, Fresnel for lenses."""

    def test_reflect_none_coating_yields_scalar(self):
        n = 3
        bulk = jnp.array([0.9, 0.85, 0.7])
        interaction = ReflectInteraction(
            reflectivity=None,
            reflectivity_scalar=bulk,
        )
        assert interaction.reflectivity is None
        assert jnp.allclose(interaction.reflectivity_scalar, bulk)

        directions = jnp.array([[0.0, 0.0, -1.0], [0.1, 0.0, -0.995], [0.0, 0.2, -0.98]])
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        points = jnp.zeros((n, 3))
        idx = jnp.array([0, 1, 2])

        _, _, coeffs, _, _ = interaction.apply(
            directions, normals, points, idx, jnp.ones(points.shape[0])
        )
        assert jnp.allclose(coeffs, bulk, atol=1e-10)

    def test_refract_none_coating_uses_fresnel(self):
        n = 2
        n_inside = jnp.array([1.5, 1.5])
        n_outside = 1.0
        trans_bulk = jnp.array([1.0, 0.9])
        interaction = RefractInteraction(
            n_inside=n_inside,
            n_outside=n_outside,
            transmittance=None,
            transmittance_scalar=trans_bulk,
        )
        assert interaction.transmittance is None

        theta = jnp.deg2rad(30.0)
        directions = jnp.tile(
            jnp.array([jnp.sin(theta), 0.0, -jnp.cos(theta)]),
            (n, 1),
        )
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        points = jnp.zeros((n, 3))
        idx = jnp.array([0, 1])

        _, _, coeffs, _, _ = interaction.apply(
            directions, normals, points, idx, jnp.ones(points.shape[0])
        )

        _, T = fresnel_unpolarized(jnp.cos(theta), n_outside, 1.5)
        expected = trans_bulk * T
        assert jnp.allclose(coeffs, expected, atol=1e-10)

    def test_slab_none_coating_uses_fresnel_squared(self):
        interaction = SlabInteraction(
            n_inside=jnp.array([1.5]),
            n_outside=1.0,
            thickness=jnp.array([0.01]),
            transmittance=None,
            transmittance_scalar=jnp.ones(1),
        )
        assert interaction.transmittance is None

        directions = jnp.array([[0.0, 0.0, -1.0]])
        normals = jnp.array([[0.0, 0.0, 1.0]])
        points = jnp.zeros((1, 3))
        idx = jnp.array([0])

        _, _, coeffs, _, _ = interaction.apply(
            directions, normals, points, idx, jnp.ones(points.shape[0])
        )
        # Normal incidence: T_face = 1 - ((1-1.5)/(1+1.5))^2 = 0.96
        # Two faces: T_total = 0.96^2 = 0.9216
        T_face = 1.0 - ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        assert jnp.allclose(coeffs, T_face**2, atol=1e-6)


class TestAngleDependentReflection:
    """End-to-end check that a coating actually changes the per-ray weight."""

    def test_tabulated_curve_changes_with_angle(self):
        # Define R(theta) dropping from 0.95 at 0 deg to 0.50 at 80 deg.
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 80.0],
            values=[0.95, 0.50],
            n_elements=1,
        )
        interaction = ReflectInteraction(
            reflectivity=coating,
            reflectivity_scalar=jnp.ones(1),
        )

        normals = jnp.array([[0.0, 0.0, 1.0]])
        points = jnp.zeros((1, 3))
        idx = jnp.array([0])

        d_normal = jnp.array([[0.0, 0.0, -1.0]])
        _, _, c_normal, _, _ = interaction.apply(
            d_normal, normals, points, idx, jnp.ones(points.shape[0])
        )

        theta = jnp.deg2rad(60.0)
        d_60 = jnp.array([[jnp.sin(theta), 0.0, -jnp.cos(theta)]])
        _, _, c_60, _, _ = interaction.apply(d_60, normals, points, idx, jnp.ones(points.shape[0]))

        theta = jnp.deg2rad(80.0)
        d_80 = jnp.array([[jnp.sin(theta), 0.0, -jnp.cos(theta)]])
        _, _, c_80, _, _ = interaction.apply(d_80, normals, points, idx, jnp.ones(points.shape[0]))

        assert jnp.allclose(c_normal, 0.95, atol=1e-6)
        assert jnp.allclose(c_80, 0.50, atol=1e-6)
        cos_80 = jnp.cos(jnp.deg2rad(80.0))
        expected = 0.95 + (0.50 - 0.95) * (1.0 - 0.5) / (1.0 - cos_80)
        assert jnp.allclose(c_60, expected, atol=1e-6)

    def test_scalar_and_curve_compose_multiplicatively(self):
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 90.0],
            values=[1.0, 0.5],
            n_elements=1,
        )
        interaction = ReflectInteraction(
            reflectivity=coating,
            reflectivity_scalar=jnp.array([0.8]),
        )
        normals = jnp.array([[0.0, 0.0, 1.0]])
        points = jnp.zeros((1, 3))
        idx = jnp.array([0])

        d = jnp.array([[0.0, 0.0, -1.0]])  # normal incidence
        _, _, c, _, _ = interaction.apply(d, normals, points, idx, jnp.ones(points.shape[0]))
        # 0.8 * 1.0 = 0.8
        assert jnp.allclose(c, 0.8, atol=1e-6)

    def test_jit_with_tabulated_reflection(self):
        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 60.0],
            values=[1.0, 0.5],
            n_elements=2,
        )
        interaction = ReflectInteraction(
            reflectivity=coating,
            reflectivity_scalar=jnp.ones(2),
        )

        @jax.jit
        def apply(it, d, n, p, idx, current_n):
            return it.apply(d, n, p, idx, current_n)

        d = jnp.array([[0.0, 0.0, -1.0], [0.5, 0.0, -0.866]])
        d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (2, 1))
        points = jnp.zeros((2, 3))
        idx = jnp.array([0, 1])

        _, _, coeffs, _, _ = apply(
            interaction,
            d,
            normals,
            points,
            idx,
            jnp.ones(2),
        )
        assert jnp.allclose(coeffs[0], 1.0, atol=1e-6)
        cos_30 = jnp.cos(jnp.deg2rad(30.0))
        expected = 1.0 + (0.5 - 1.0) * (1.0 - cos_30) / (1.0 - 0.5)
        assert jnp.allclose(coeffs[1], expected, atol=1e-6)

    def test_tabulated_reproduces_fresnel(self):
        """A coating sampled from Fresnel matches the bare-interface path."""
        n_in = 1.5
        n_out = 1.0
        angles_deg = jnp.linspace(0.0, 89.0, 50)
        cos_i = jnp.cos(jnp.deg2rad(angles_deg))
        _, T_table = fresnel_unpolarized(cos_i, n_out, n_in)
        coating = TabulatedCoating.from_degrees(
            angles_deg=[float(x) for x in angles_deg],
            values=[float(x) for x in T_table],
            n_elements=1,
        )

        cos_i_q = jnp.array([jnp.cos(jnp.deg2rad(45.0))])
        T_from_table = coating(cos_i_q, jnp.array([0]))
        _, T_from_fresnel = fresnel_unpolarized(cos_i_q, n_out, n_in)
        assert jnp.allclose(T_from_table, T_from_fresnel, atol=1e-3)


class TestFactoryFlow:
    """``mirror_group``/``refractive_group`` accept an optional ``coating=``."""

    def test_mirror_group_accepts_coating(self):
        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.mirrors import mirror_group

        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 90.0],
            values=[0.95, 0.5],
            n_elements=2,
        )
        g = mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([1.0, 1.0]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 0)),
            offsets=jnp.zeros((2, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.05, 0.05]),
                inner_radii=jnp.zeros(2),
            ),
            reflectivity=0.95,
            coating=coating,
            sample_key=jax.random.key(0),
        )
        assert isinstance(g.interaction_module.reflectivity, TabulatedCoating)
        assert jnp.allclose(
            g.interaction_module.reflectivity_scalar,
            jnp.full(2, 0.95),
        )

    def test_mirror_group_no_coating(self):
        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.mirrors import mirror_group

        g = mirror_group(
            positions=jnp.zeros((1, 3)),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([1.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.05]),
                inner_radii=jnp.zeros(1),
            ),
            reflectivity=0.9,
            sample_key=jax.random.key(0),
        )
        # No coating -> reflectivity is None on the interaction.
        assert g.interaction_module.reflectivity is None
        assert jnp.allclose(g.interaction_module.reflectivity_scalar, 0.9)

    def test_refractive_group_no_coating_defaults_to_fresnel(self):
        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.lenses import refractive_group

        g = refractive_group(
            positions=jnp.array([[0.0, 0.0, 0.1]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([5.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.02]),
                inner_radii=jnp.zeros(1),
            ),
            n_inside=jnp.array([1.5]),
            n_outside=1.0,
            sample_key=jax.random.key(0),
        )
        assert g.interaction_module.transmittance is None
        assert jnp.allclose(
            g.interaction_module.transmittance_scalar,
            jnp.ones(1),
        )

    def test_refractive_group_with_coating(self):
        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.lenses import refractive_group

        coating = TabulatedCoating.from_degrees(
            angles_deg=[0.0, 90.0],
            values=[0.99, 0.0],
            n_elements=1,
        )
        g = refractive_group(
            positions=jnp.array([[0.0, 0.0, 0.1]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([5.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.02]),
                inner_radii=jnp.zeros(1),
            ),
            n_inside=jnp.array([1.5]),
            n_outside=1.0,
            transmittance=1.0,
            coating=coating,
            sample_key=jax.random.key(0),
        )
        assert isinstance(g.interaction_module.transmittance, TabulatedCoating)
        assert jnp.allclose(
            g.interaction_module.transmittance_scalar,
            jnp.array([1.0]),
        )
