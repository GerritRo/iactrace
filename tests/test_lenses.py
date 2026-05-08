import jax
import jax.numpy as jnp

from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import InteractionType, RefractInteraction, SlabInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import AsphericSurfaceGroup


class TestAsphericDiskLensGroup:
    """Test OpticalElementGroup configured as a curved refractive lens."""

    def _make_lens(self, positions, rotations, curvatures, conics, aspherics,
                   radii, n_inside, n_outside=1.0, transmittance=None):
        """Helper to build an OpticalElementGroup with RefractInteraction."""
        n = curvatures.shape[0]
        surface = AsphericSurfaceGroup(
            curvatures=curvatures,
            conics=conics,
            aspherics=aspherics,
            offsets=jnp.zeros((n, 2)),
        )
        aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
        n_inside_arr = jnp.broadcast_to(jnp.asarray(n_inside, dtype=float), (n,))
        if transmittance is None:
            transmittance_arr = jnp.ones(n)
        else:
            transmittance_arr = jnp.broadcast_to(jnp.asarray(transmittance, dtype=float), (n,))
        interaction = RefractInteraction(
            n_inside=n_inside_arr,
            n_outside=float(n_outside),
            transmittance=transmittance_arr,
        )
        return OpticalElementGroup(
            positions=positions,
            rotations=rotations,
            surface=surface,
            aperture=aperture,
            interaction_module=interaction,
            sample_key=jax.random.key(0),
            optical_stage=0,
        )

    def test_basic_creation(self):
        """Lens group can be created with valid parameters."""
        lens = self._make_lens(
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
        assert jnp.allclose(lens.interaction_module.n_inside, jnp.array([1.5]))
        assert lens.interaction_module.n_outside == 1.0

    def test_refractive_index_broadcast(self):
        """Scalar n_inside broadcasts to all elements."""
        lens = self._make_lens(
            positions=jnp.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([0.1, 0.2]),
            conics=jnp.array([0.0, 0.0]),
            aspherics=jnp.zeros((2, 2)),
            radii=jnp.array([0.5, 0.5]),
            n_inside=1.5,
        )

        assert lens.interaction_module.n_inside.shape == (2,)
        assert jnp.allclose(lens.interaction_module.n_inside, jnp.array([1.5, 1.5]))

    def test_check_aperture(self):
        """Aperture check validates points against circular aperture."""
        lens = self._make_lens(
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
        lens = self._make_lens(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([0.1]),
            conics=jnp.array([0.0]),
            aspherics=jnp.zeros((1, 2)),
            radii=jnp.array([0.5]),
            n_inside=1.5,
        )

        assert jnp.allclose(lens.interaction_module.transmittance, jnp.array([1.0]))


class TestPlanoSlabGroup:
    """Test OpticalElementGroup configured as a flat parallel-sided slab."""

    def _make_slab(self, positions, rotations, radii, thickness, n_inside,
                   n_outside=1.0, transmittance=None):
        """Helper to build an OpticalElementGroup with SlabInteraction."""
        n = radii.shape[0]
        surface = AsphericSurfaceGroup(
            curvatures=jnp.zeros(n),
            conics=jnp.zeros(n),
            aspherics=jnp.zeros((n, 1)),
            offsets=jnp.zeros((n, 2)),
        )
        aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
        n_inside_arr = jnp.broadcast_to(jnp.asarray(n_inside, dtype=float), (n,))
        thickness_arr = jnp.broadcast_to(jnp.asarray(thickness, dtype=float), (n,))
        if transmittance is None:
            transmittance_arr = jnp.ones(n)
        else:
            transmittance_arr = jnp.broadcast_to(jnp.asarray(transmittance, dtype=float), (n,))
        interaction = SlabInteraction(
            n_inside=n_inside_arr,
            n_outside=float(n_outside),
            thickness=thickness_arr,
            transmittance=transmittance_arr,
        )
        return OpticalElementGroup(
            positions=positions,
            rotations=rotations,
            surface=surface,
            aperture=aperture,
            interaction_module=interaction,
            sample_key=jax.random.key(0),
            optical_stage=0,
        )

    def test_basic_creation(self):
        """Slab group can be created with valid parameters."""
        slab = self._make_slab(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            radii=jnp.array([0.5]),
            thickness=0.01,
            n_inside=1.5,
        )

        assert len(slab) == 1
        assert slab.interaction == InteractionType.SLAB
        assert jnp.allclose(slab.interaction_module.n_inside, jnp.array([1.5]))
        assert jnp.allclose(slab.interaction_module.thickness, jnp.array([0.01]))

    def test_thickness_broadcast(self):
        """Scalar thickness broadcasts to all elements."""
        slab = self._make_slab(
            positions=jnp.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
            rotations=jnp.zeros((2, 3)),
            radii=jnp.array([0.5, 0.5]),
            thickness=0.02,
            n_inside=1.5,
        )

        assert slab.interaction_module.thickness.shape == (2,)
        assert jnp.allclose(slab.interaction_module.thickness, jnp.array([0.02, 0.02]))

    def test_flat_surface_parameters(self):
        """Slab has zero curvature, conic, and aspherics."""
        slab = self._make_slab(
            positions=jnp.array([[0.0, 0.0, 5.0]]),
            rotations=jnp.zeros((1, 3)),
            radii=jnp.array([0.5]),
            thickness=0.01,
            n_inside=1.5,
        )

        assert jnp.allclose(slab.surface.curvatures, jnp.array([0.0]))
        assert jnp.allclose(slab.surface.conics, jnp.array([0.0]))

    def test_check_aperture(self):
        """Aperture check works for slab."""
        slab = self._make_slab(
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
