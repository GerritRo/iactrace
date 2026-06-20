import jax.numpy as jnp

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
        normal = jnp.array([0.0, 0.0, 1.0])      # Pointing up
        n1, n2 = 1.0, 1.5

        refracted, cos_i, tir = refract(direction, normal, n1, n2)

        # Direction should be unchanged (still pointing down)
        assert jnp.allclose(refracted, direction, atol=1e-10)
        assert jnp.isclose(cos_i, 1.0, atol=1e-10)
        assert not tir

    def test_snells_law_air_to_glass(self):
        """Verify Snell's law: n1*sin(theta1) = n2*sin(theta2)."""
        theta_i = jnp.pi / 4
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        n1, n2 = 1.0, 1.5

        refracted, _, tir = refract(direction, normal, n1, n2)

        sin_theta_i = jnp.sin(theta_i)
        cos_t = _cos_t_from_direction(refracted, normal)
        sin_theta_t = jnp.sqrt(1.0 - cos_t**2)
        assert jnp.isclose(n1 * sin_theta_i, n2 * sin_theta_t, rtol=1e-6)
        assert not tir

    def test_snells_law_glass_to_air(self):
        """Refraction from glass to air bends ray away from normal."""
        theta_i = jnp.deg2rad(20.0)
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        n1, n2 = 1.5, 1.0

        refracted, _, tir = refract(direction, normal, n1, n2)

        sin_theta_i = jnp.sin(theta_i)
        cos_t = _cos_t_from_direction(refracted, normal)
        sin_theta_t = jnp.sqrt(1.0 - cos_t**2)
        assert jnp.isclose(n1 * sin_theta_i, n2 * sin_theta_t, rtol=1e-6)
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

    def test_refracted_direction_is_normalized(self):
        """Output direction should be unit vector."""
        direction = jnp.array([0.3, 0.2, -0.9])
        direction = direction / jnp.linalg.norm(direction)
        normal = jnp.array([0.0, 0.0, 1.0])

        refracted, _, _ = refract(direction, normal, 1.0, 1.5)

        assert jnp.isclose(jnp.linalg.norm(refracted), 1.0, atol=1e-10)

    def test_equal_indices_no_change(self):
        """When n1 == n2, ray direction is unchanged."""
        direction = jnp.array([0.3, 0.2, -0.9])
        direction = direction / jnp.linalg.norm(direction)
        normal = jnp.array([0.0, 0.0, 1.0])

        refracted, _, tir = refract(direction, normal, 1.5, 1.5)

        assert jnp.allclose(refracted, direction, atol=1e-10)
        assert not tir

    def test_refraction_plane_preserved(self):
        """Refracted ray stays in the plane of incidence."""
        direction = jnp.array([0.5, 0.0, -0.866])
        direction = direction / jnp.linalg.norm(direction)
        normal = jnp.array([0.0, 0.0, 1.0])

        refracted, _, _ = refract(direction, normal, 1.0, 1.5)

        assert jnp.isclose(refracted[1], 0.0, atol=1e-10)


class TestFresnelUnpolarized:
    """Test Fresnel reflection/transmission coefficients.

    ``fresnel_unpolarized`` derives ``cos_t`` from ``cos_i`` via Snell,
    so the API takes only the incidence cosine and the two indices.
    """

    def test_normal_incidence_formula(self):
        """At normal incidence, R = ((n1-n2)/(n1+n2))^2."""
        n1, n2 = 1.0, 1.5
        R, T = fresnel_unpolarized(1.0, n1, n2)

        R_expected = ((n1 - n2) / (n1 + n2))**2
        assert jnp.isclose(R, R_expected, rtol=1e-6)
        assert jnp.isclose(R + T, 1.0, atol=1e-10)

    def test_energy_conservation(self):
        """R + T = 1 for all angles below TIR."""
        n1, n2 = 1.0, 1.5
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
            direction, normal, position, n_out, n_in, thickness,
        )

        assert jnp.allclose(exit_dir, direction, atol=1e-10)
        assert jnp.isclose(exit_pos[2], -thickness, atol=1e-10)
        assert jnp.isclose(exit_pos[0], 0.0, atol=1e-10)
        assert jnp.isclose(exit_pos[1], 0.0, atol=1e-10)
        assert jnp.isclose(cos_i, 1.0, atol=1e-10)
        assert valid
        # Normal incidence: path inside slab = thickness; OPL = n_in * thickness.
        assert jnp.isclose(path_length, thickness, atol=1e-10)

    def test_oblique_incidence_direction_preserved(self):
        """For parallel surfaces, exit direction equals entry direction."""
        theta_i = jnp.deg2rad(30.0)
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        exit_dir, _, cos_i, valid, path_length = refract_slab(
            direction, normal, position, n_out, n_in, thickness,
        )

        assert jnp.allclose(exit_dir, direction, atol=1e-6)
        assert jnp.isclose(cos_i, jnp.cos(theta_i), atol=1e-6)
        assert valid
        # Oblique incidence: path_length > thickness because the ray is
        # tilted relative to the slab normal inside the glass.
        assert path_length > thickness

    def test_lateral_displacement_increases_with_angle(self):
        """Oblique rays have larger lateral offset than normal rays."""
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        _, pos_normal, *_ = refract_slab(
            jnp.array([0.0, 0.0, -1.0]),
            normal, position, n_out, n_in, thickness,
        )

        theta_30 = jnp.deg2rad(30.0)
        dir_30 = jnp.array([jnp.sin(theta_30), 0.0, -jnp.cos(theta_30)])
        _, pos_30, *_ = refract_slab(
            dir_30, normal, position, n_out, n_in, thickness,
        )

        assert jnp.abs(pos_30[0]) > jnp.abs(pos_normal[0]) + 1e-10

    def test_slab_fresnel_squared_at_normal_incidence(self):
        """Caller computes T = T_face^2 from the cos_i returned by refract_slab."""
        direction = jnp.array([0.0, 0.0, -1.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        _, _, cos_i, valid, _ = refract_slab(
            direction, normal, position, n_out, n_in, thickness,
        )
        _, T_face = fresnel_unpolarized(cos_i, n_out, n_in)
        transmittance = T_face * T_face

        R_single = ((n_out - n_in) / (n_out + n_in))**2
        T_expected = (1 - R_single)**2

        assert jnp.isclose(transmittance, T_expected, rtol=1e-4)
        assert valid

    def test_slab_high_angle_no_tir(self):
        """No TIR for moderate angles in typical glass."""
        theta_i = jnp.deg2rad(40.0)
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.01

        _, _, _, valid, _ = refract_slab(
            direction, normal, position, n_out, n_in, thickness,
        )
        assert valid

    def test_slab_thickness_affects_offset(self):
        """Thicker slab produces larger lateral offset."""
        theta_i = jnp.deg2rad(30.0)
        direction = jnp.array([jnp.sin(theta_i), 0.0, -jnp.cos(theta_i)])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5

        _, pos_thin, *_ = refract_slab(
            direction, normal, position, n_out, n_in, thickness=0.01,
        )
        _, pos_thick, *_ = refract_slab(
            direction, normal, position, n_out, n_in, thickness=0.02,
        )

        assert jnp.abs(pos_thick[0]) > jnp.abs(pos_thin[0])

    def test_slab_z_displacement_equals_thickness(self):
        """Z displacement through slab equals thickness at normal incidence."""
        direction = jnp.array([0.0, 0.0, -1.0])
        normal = jnp.array([0.0, 0.0, 1.0])
        position = jnp.array([0.0, 0.0, 0.0])
        n_out, n_in = 1.0, 1.5
        thickness = 0.015

        _, exit_pos, *_ = refract_slab(
            direction, normal, position, n_out, n_in, thickness,
        )
        assert jnp.isclose(exit_pos[2] - position[2], -thickness, atol=1e-10)


class TestReflect:
    """Test reflection function."""

    def test_normal_incidence_reverses_direction(self):
        """Ray normal to surface reflects back along same line."""
        from iactrace.core.interactions import reflect

        direction = jnp.array([0.0, 0.0, -1.0])
        normal = jnp.array([0.0, 0.0, 1.0])

        reflected, cos_angle = reflect(direction, normal)

        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(reflected, expected, atol=1e-10)
        assert jnp.isclose(cos_angle.squeeze(), 1.0, atol=1e-10)

    def test_45_degree_reflection(self):
        """45 degree incidence on horizontal surface."""
        from iactrace.core.interactions import reflect

        # Ray coming from upper left, hitting horizontal surface
        direction = jnp.array([1.0, 0.0, -1.0]) / jnp.sqrt(2)
        normal = jnp.array([0.0, 0.0, 1.0])

        reflected, cos_angle = reflect(direction, normal)

        # Should reflect to upper right
        expected = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2)
        assert jnp.allclose(reflected, expected, atol=1e-10)
        assert jnp.isclose(cos_angle.squeeze(), 1.0 / jnp.sqrt(2), atol=1e-10)

    def test_reflected_direction_is_unit_length(self):
        """Reflected direction should be unit vector."""
        from iactrace.core.interactions import reflect

        direction = jnp.array([0.3, 0.4, -0.866])
        direction = direction / jnp.linalg.norm(direction)
        normal = jnp.array([0.0, 0.0, 1.0])

        reflected, _ = reflect(direction, normal)

        assert jnp.isclose(jnp.linalg.norm(reflected), 1.0, atol=1e-10)
