import jax
import jax.numpy as jnp
import pytest

from iactrace.core.interactions import (
    ReflectInteraction,
    RefractInteraction,
    SlabInteraction,
)
from iactrace.core.refractive_index import ConstantIndex
from iactrace.core.responses import (
    ConstantResponse,
    TabulatedResponse,
    fresnel_unpolarized,
)


class TestConstantResponse:
    """Constant value."""

    def test_returns_per_element_value_ignoring_angle(self):
        coating = ConstantResponse(values=jnp.array([0.9, 0.85, 0.7]))
        idx = jnp.array([0, 1, 2, 1])
        cos = jnp.array([1.0, 0.5, 0.1, 0.9])  # varied angles are ignored
        assert jnp.allclose(coating(cos, idx), jnp.array([0.9, 0.85, 0.7, 0.85]))
        # same element -> same value regardless of incidence cosine
        for c in (1.0, 0.5, 0.0):
            assert jnp.allclose(coating(jnp.full(3, c), jnp.array([0, 0, 0])), 0.9)


class TestTabulatedResponse:
    """Linear interpolation over a precomputed cos(angle) table."""

    def test_from_degrees_per_element(self):
        per_elem = jnp.array([[1.0, 0.7, 0.2], [0.9, 0.5, 0.1], [0.95, 0.6, 0.15]])
        coating = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=per_elem,
            n_elements=3,
        )
        # (N, Kc, Kw); angle-only curve is the degenerate single-wavelength grid.
        assert coating.values.shape == (3, 3, 1)

    def test_from_degrees_rejects_wrong_n(self):
        with pytest.raises(ValueError, match="must match"):
            TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0],
                values=jnp.array([[1.0, 0.2], [0.9, 0.1]]),
                n_elements=3,
            )

    def test_exact_at_knots_and_linear_between(self):
        # Curve: R(0 deg) = 0.9, R(60 deg) = 0.4 (cos = 0.5); exact at both knots.
        coating = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 60.0], values=[0.9, 0.4], n_elements=1
        )
        assert jnp.allclose(coating(jnp.array([1.0]), jnp.array([0])), 0.9, atol=1e-6)
        assert jnp.allclose(coating(jnp.array([0.5]), jnp.array([0])), 0.4, atol=1e-6)
        # Linear in cos between knots: midpoint of a 1.0 -> 0.0 ramp is 0.5.
        ramp = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0], values=[1.0, 0.0], n_elements=1
        )
        assert jnp.allclose(ramp(jnp.array([0.5]), jnp.array([0])), 0.5, atol=1e-6)

    def test_unsorted_input_normalized(self):
        c1 = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 45.0, 90.0],
            values=[1.0, 0.7, 0.2],
            n_elements=1,
        )
        c2 = TabulatedResponse.from_degrees(
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
            reflectivity_curve=None,
            reflectivity=bulk,
        )
        assert interaction.reflectivity_curve is None
        assert jnp.allclose(interaction.reflectivity, bulk)

        directions = jnp.array([[0.0, 0.0, -1.0], [0.1, 0.0, -0.995], [0.0, 0.2, -0.98]])
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        points = jnp.zeros((n, 3))
        idx = jnp.array([0, 1, 2])

        _, _, coeffs, _, _ = interaction.apply(
            directions, normals, points, idx, jnp.ones(points.shape[0])
        )
        assert jnp.allclose(coeffs, bulk, atol=1e-10)

    def test_transmissive_none_coating_uses_fresnel(self):
        """With ``transmittance_curve=None`` a refracting interface applies single-face
        Fresnel (scaled by the bulk), and a slab applies it squared (two faces)."""
        # RefractInteraction: single-face Fresnel at 30 deg, times the bulk scalar.
        trans_bulk = jnp.array([1.0, 0.9])
        refract_it = RefractInteraction(
            index=ConstantIndex(jnp.array([1.5, 1.5])),
            transmittance_curve=None,
            transmittance=trans_bulk,
        )
        assert refract_it.transmittance_curve is None
        theta = jnp.deg2rad(30.0)
        directions = jnp.tile(jnp.array([jnp.sin(theta), 0.0, -jnp.cos(theta)]), (2, 1))
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (2, 1))
        _, _, coeffs, _, _ = refract_it.apply(
            directions, normals, jnp.zeros((2, 3)), jnp.array([0, 1]), jnp.ones(2)
        )
        _, T = fresnel_unpolarized(jnp.cos(theta), 1.0, 1.5)
        assert jnp.allclose(coeffs, trans_bulk * T, atol=1e-10)

        # SlabInteraction: two-face Fresnel (T_face^2) at normal incidence.
        slab_it = SlabInteraction(
            index=ConstantIndex(jnp.array([1.5])),
            thickness=jnp.array([0.01]),
            transmittance_curve=None,
            transmittance=jnp.ones(1),
        )
        assert slab_it.transmittance_curve is None
        _, _, coeffs, _, _ = slab_it.apply(
            jnp.array([[0.0, 0.0, -1.0]]),
            jnp.array([[0.0, 0.0, 1.0]]),
            jnp.zeros((1, 3)),
            jnp.array([0]),
            jnp.ones(1),
        )
        T_face = 1.0 - ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        assert jnp.allclose(coeffs, T_face**2, atol=1e-6)


class TestAngleDependentReflection:
    """End-to-end check that a coating actually changes the per-ray weight."""

    def test_tabulated_curve_changes_with_angle(self):
        # Define R(theta) dropping from 0.95 at 0 deg to 0.50 at 80 deg.
        coating = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 80.0],
            values=[0.95, 0.50],
            n_elements=1,
        )
        interaction = ReflectInteraction(
            reflectivity_curve=coating,
            reflectivity=jnp.ones(1),
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
        coating = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 90.0],
            values=[1.0, 0.5],
            n_elements=1,
        )
        interaction = ReflectInteraction(
            reflectivity_curve=coating,
            reflectivity=jnp.array([0.8]),
        )
        normals = jnp.array([[0.0, 0.0, 1.0]])
        points = jnp.zeros((1, 3))
        idx = jnp.array([0])

        d = jnp.array([[0.0, 0.0, -1.0]])  # normal incidence
        _, _, c, _, _ = interaction.apply(d, normals, points, idx, jnp.ones(points.shape[0]))
        # 0.8 * 1.0 = 0.8
        assert jnp.allclose(c, 0.8, atol=1e-6)

    def test_jit_with_tabulated_reflection(self):
        coating = TabulatedResponse.from_degrees(
            angles_deg=[0.0, 60.0],
            values=[1.0, 0.5],
            n_elements=2,
        )
        interaction = ReflectInteraction(
            reflectivity_curve=coating,
            reflectivity=jnp.ones(2),
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


class TestFactoryFlow:
    """Factories take the ``X`` / ``X_curve`` pair, in the same shape everywhere."""

    def test_mirror_and_refractive_group_accept_curves(self):
        """reflectivity_curve lands on the mirror interaction and
        transmittance_curve on the lens one, each keeping its bulk scalar."""
        from iactrace.core.apertures import DiskAperture
        from iactrace.telescope.lenses import refractive_group
        from iactrace.telescope.mirrors import mirror_group

        mirror = mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([1.0, 1.0]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 0)),
            offsets=jnp.zeros((2, 2)),
            aperture=DiskAperture(radii=jnp.array([0.05, 0.05]), inner_radii=jnp.zeros(2)),
            reflectivity=0.95,
            reflectivity_curve=TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0], values=[0.95, 0.5], n_elements=2
            ),
            sample_key=jax.random.key(0),
        )
        assert isinstance(mirror.interaction_module.reflectivity_curve, TabulatedResponse)
        assert jnp.allclose(mirror.interaction_module.reflectivity, jnp.full(2, 0.95))

        lens = refractive_group(
            positions=jnp.array([[0.0, 0.0, 0.1]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([5.0]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(radii=jnp.array([0.02]), inner_radii=jnp.zeros(1)),
            index=jnp.array([1.5]),
            transmittance=1.0,
            transmittance_curve=TabulatedResponse.from_degrees(
                angles_deg=[0.0, 90.0], values=[0.99, 0.0], n_elements=1
            ),
            sample_key=jax.random.key(0),
        )
        assert isinstance(lens.interaction_module.transmittance_curve, TabulatedResponse)
        assert jnp.allclose(lens.interaction_module.transmittance, jnp.array([1.0]))
