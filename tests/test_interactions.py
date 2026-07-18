import jax.numpy as jnp
import pytest

from iactrace.core.coatings import fresnel_unpolarized
from iactrace.core.interactions import refract, refract_slab


def _cos_t_from_direction(refracted, normal):
    """Derive the transmitted-angle cosine from the refracted ray direction."""
    return jnp.abs(jnp.dot(refracted, normal))


class TestRefract:
    """Test single-surface refraction (Snell's law)."""

    def test_normal_incidence_no_deviation(self):
        """Ray normal to surface passes straight through."""
        direction = jnp.array([0.0, 0.0, -1.0])  # Pointing down
        normal = jnp.array([0.0, 0.0, 1.0])  # Pointing up
        n1, n2 = 1.0, 1.5

        refracted, cos_i, tir = refract(direction, normal, n1, n2)

        # Direction should be unchanged (still pointing down)
        assert jnp.allclose(refracted, direction, atol=1e-10)
        assert jnp.isclose(cos_i, 1.0, atol=1e-10)
        assert not tir

    @pytest.mark.parametrize(
        ("theta_i", "n1", "n2"),
        [(jnp.pi / 4, 1.0, 1.5), (jnp.deg2rad(20.0), 1.5, 1.0)],
    )
    def test_snells_law(self, theta_i, n1, n2):
        """Snell's law holds entering and leaving glass: n1 sinθ1 = n2 sinθ2."""
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])

        refracted, _, tir = refract(direction, normal, n1, n2)

        cos_t = _cos_t_from_direction(refracted, normal)
        sin_theta_t = jnp.sqrt(1.0 - cos_t**2)
        assert jnp.isclose(n1 * jnp.sin(theta_i), n2 * sin_theta_t, rtol=1e-6)
        assert not tir

    def test_total_internal_reflection_critical_angle(self):
        """TIR occurs above critical angle when going from dense to less dense."""
        n1, n2 = 1.5, 1.0
        theta_critical = jnp.arcsin(n2 / n1)

        theta_below = theta_critical - 0.05
        direction_below = jnp.array([jnp.sin(theta_below), 0.0, -jnp.cos(theta_below)])
        normal = jnp.array([0.0, 0.0, 1.0])
        _, _, tir_below = refract(direction_below, normal, n1, n2)
        assert not tir_below

        theta_above = theta_critical + 0.05
        direction_above = jnp.array([jnp.sin(theta_above), 0.0, -jnp.cos(theta_above)])
        _, _, tir_above = refract(direction_above, normal, n1, n2)
        assert tir_above


class TestFresnelUnpolarized:
    """Test Fresnel reflection/transmission coefficients.

    ``fresnel_unpolarized`` derives ``cos_t`` from ``cos_i`` via Snell,
    so the API takes only the incidence cosine and the two indices.
    """

    def test_energy_conservation_and_normal_formula(self):
        """R + T = 1 for all angles below TIR, and at normal incidence
        R = ((n1-n2)/(n1+n2))^2."""
        n1, n2 = 1.0, 1.5
        R0, _ = fresnel_unpolarized(1.0, n1, n2)
        assert jnp.isclose(R0, ((n1 - n2) / (n1 + n2)) ** 2, rtol=1e-6)
        for theta_i in [0.0, 0.2, 0.5, 0.8, 1.0]:
            R, T = fresnel_unpolarized(jnp.cos(theta_i), n1, n2)
            assert jnp.isclose(R + T, 1.0, atol=1e-10)

    def test_reflectance_increases_with_angle(self):
        """Reflectance grows monotonically with incidence angle."""
        n1, n2 = 1.0, 1.5
        R_values = [
            float(fresnel_unpolarized(jnp.cos(jnp.deg2rad(theta)), n1, n2)[0])
            for theta in [0, 30, 60, 80]
        ]
        for i in range(len(R_values) - 1):
            assert R_values[i] <= R_values[i + 1] + 1e-10

    def test_symmetric_indices_same_reflectance(self):
        """R is the same from either side at corresponding angles (Stokes)."""
        n1, n2 = 1.0, 1.5
        theta_1 = jnp.deg2rad(30.0)
        sin_2 = (n1 / n2) * jnp.sin(theta_1)
        theta_2 = jnp.arcsin(sin_2)

        R_12, _ = fresnel_unpolarized(jnp.cos(theta_1), n1, n2)
        R_21, _ = fresnel_unpolarized(jnp.cos(theta_2), n2, n1)

        assert jnp.isclose(R_12, R_21, rtol=1e-6)

    def test_tir_returns_unit_reflectance(self):
        """Past the critical angle, R = 1 and T = 0."""
        n1, n2 = 1.5, 1.0
        theta_above = jnp.arcsin(n2 / n1) + 0.05
        R, T = fresnel_unpolarized(jnp.cos(theta_above), n1, n2)
        assert jnp.isclose(R, 1.0, atol=1e-6)
        assert jnp.isclose(T, 0.0, atol=1e-6)


class TestRefractSlab:
    """Test parallel-sided slab refraction (pure geometry).

    ``refract_slab`` returns the 4-tuple
    ``(exit_direction, exit_position, cos_i, valid)``. The Fresnel
    transmittance, when needed, is computed from ``cos_i`` by the caller.
    """

    def test_normal_incidence_straight_through(self):
        """Ray normal to slab exits with same direction."""
        direction = jnp.array([0.0, 0.0, -1.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        exit_dir, exit_pos, cos_i, valid, path_length = refract_slab(
            direction,
            normal,
            position,
            n_out,
            n_in,
            thickness,
        )

        assert jnp.allclose(exit_dir, direction, atol=1e-10)
        assert jnp.isclose(exit_pos[2], -thickness, atol=1e-10)
        assert jnp.isclose(exit_pos[0], 0.0, atol=1e-10)
        assert jnp.isclose(exit_pos[1], 0.0, atol=1e-10)
        assert jnp.isclose(cos_i, 1.0, atol=1e-10)
        assert valid
        # Normal incidence: path inside slab = thickness; OPL = n_in * thickness.
        assert jnp.isclose(path_length, thickness, atol=1e-10)
        # The caller's two-face transmittance T = T_face^2 = (1 - R_single)^2.
        _, T_face = fresnel_unpolarized(cos_i, n_out, n_in)
        R_single = ((n_out - n_in) / (n_out + n_in)) ** 2
        assert jnp.isclose(T_face * T_face, (1 - R_single) ** 2, rtol=1e-4)

    def test_oblique_incidence_direction_preserved(self):
        """For parallel surfaces, exit direction equals entry direction."""
        theta_i = jnp.deg2rad(30.0)
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        exit_dir, _, cos_i, valid, path_length = refract_slab(
            direction,
            normal,
            position,
            n_out,
            n_in,
            thickness,
        )

        assert jnp.allclose(exit_dir, direction, atol=1e-6)
        assert jnp.isclose(cos_i, jnp.cos(theta_i), atol=1e-6)
        assert valid
        # Oblique incidence: path_length > thickness because the ray is
        # tilted relative to the slab normal inside the glass.
        assert path_length > thickness

    def test_lateral_offset_grows_with_angle_and_thickness(self):
        """Slab lateral offset increases with both incidence angle and thickness."""
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5

        def offset(theta_deg, thickness):
            theta = jnp.deg2rad(theta_deg)
            direction = jnp.array([jnp.sin(theta), 0.0, -jnp.cos(theta)])
            _, pos, *_ = refract_slab(direction, normal, position, n_out, n_in, thickness)
            return jnp.abs(pos[0])

        assert offset(30.0, 0.01) > offset(0.0, 0.01) + 1e-10  # grows with angle
        assert offset(30.0, 0.02) > offset(30.0, 0.01)  # grows with thickness


class TestReflect:
    """Test reflection function."""

    def test_reflection_normal_and_oblique(self):
        """Reflection reverses a normal-incidence ray and mirrors a 45 deg ray."""
        from iactrace.core.interactions import reflect

        # Normal incidence: reflects straight back, cos = 1.
        r, cos_angle = reflect(jnp.array([0.0, 0.0, -1.0]), jnp.array([0.0, 0.0, 1.0]))
        assert jnp.allclose(r, jnp.array([0.0, 0.0, 1.0]), atol=1e-10)
        assert jnp.isclose(cos_angle.squeeze(), 1.0, atol=1e-10)

        # 45 deg incidence from upper left -> reflects to upper right, cos = 1/sqrt2.
        r, cos_angle = reflect(
            jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2), jnp.array([0.0, 0.0, 1.0])
        )
        assert jnp.allclose(r, jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2), atol=1e-10)
        assert jnp.isclose(cos_angle.squeeze(), 1.0 / jnp.sqrt(2), atol=1e-10)
