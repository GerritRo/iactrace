import jax.numpy as jnp

from iactrace.core.optics import InteractionType
from iactrace.telescope.lenses import AsphericDiskLensGroup, PlanoSlabGroup


class TestAsphericDiskLensGroup:
    """Test AsphericDiskLensGroup for curved refractive surfaces."""

    def test_basic_creation(self):
        """Lens group can be created with valid parameters."""
        lens = AsphericDiskLensGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.array([[0.0, 0.0, 0.0]]),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 2)),
            radii=jnp.array([0.5]),
            n_inside=1.5,
        )

        assert len(lens) == 1
        assert lens.interaction == InteractionType.REFRACT
        assert jnp.allclose(lens.n_inside, jnp.array([1.5]))
        assert lens.n_outside == 1.0

    def test_refractive_index_broadcast(self):
        """Scalar n_inside broadcasts to all elements."""
        lens = AsphericDiskLensGroup(
            positions=jnp.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.1, 0.2]),
            conics=jnp.array([0.0, 0.0]),
            aspherics=jnp.zeros((2, 2)),
            radii=jnp.array([0.5, 0.5]),
            n_inside=1.5,  # Scalar
        )

        assert lens.n_inside.shape == (2,)
        assert jnp.allclose(lens.n_inside, jnp.array([1.5, 1.5]))

    def test_check_aperture(self):
        """Aperture check validates points against circular aperture."""
        lens = AsphericDiskLensGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 2)),
            radii=jnp.array([0.5]),
            n_inside=1.5,
        )

        # Inside aperture
        assert lens.check_aperture(0.0, 0.0, 0)
        assert lens.check_aperture(0.3, 0.0, 0)
        assert lens.check_aperture(0.3, 0.3, 0)

        # Outside aperture
        assert not lens.check_aperture(0.6, 0.0, 0)
        assert not lens.check_aperture(0.4, 0.4, 0)

    def test_transmittance_defaults_to_one(self):
        """Transmittance defaults to 1.0 if not specified."""
        lens = AsphericDiskLensGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 2)),
            radii=jnp.array([0.5]),
            n_inside=1.5,
        )

        assert jnp.allclose(lens.transmittance, jnp.array([1.0]))


class TestPlanoSlabGroup:
    """Test PlanoSlabGroup for flat parallel-sided windows."""

    def test_basic_creation(self):
        """Slab group can be created with valid parameters."""
        slab = PlanoSlabGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            radii=jnp.array([0.5]),
            thickness=0.01,
            n_inside=1.5,
        )

        assert len(slab) == 1
        assert slab.interaction == InteractionType.SLAB
        assert jnp.allclose(slab.n_inside, jnp.array([1.5]))
        assert jnp.allclose(slab.thickness, jnp.array([0.01]))

    def test_thickness_broadcast(self):
        """Scalar thickness broadcasts to all elements."""
        slab = PlanoSlabGroup(
            positions=jnp.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
            rotations=jnp.zeros((2, 3)),
            radii=jnp.array([0.5, 0.5]),
            thickness=0.02,  # Scalar
            n_inside=1.5,
        )

        assert slab.thickness.shape == (2,)
        assert jnp.allclose(slab.thickness, jnp.array([0.02, 0.02]))

    def test_flat_surface_parameters(self):
        """Slab has zero curvature, conic, and aspherics."""
        slab = PlanoSlabGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            radii=jnp.array([0.5]),
            thickness=0.01,
            n_inside=1.5,
        )

        assert jnp.allclose(slab.curvatures, jnp.array([0.0]))
        assert jnp.allclose(slab.conics, jnp.array([0.0]))

    def test_check_aperture(self):
        """Aperture check works for slab."""
        slab = PlanoSlabGroup(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            radii=jnp.array([0.3]),
            thickness=0.01,
            n_inside=1.5,
        )

        # Inside aperture
        assert slab.check_aperture(0.0, 0.0, 0)
        assert slab.check_aperture(0.2, 0.0, 0)

        # Outside aperture
        assert not slab.check_aperture(0.4, 0.0, 0)
