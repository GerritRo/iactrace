"""Tests for the Python-friendly telescope component factories.

Covers :mod:`iactrace.telescope.mirrors`, :mod:`iactrace.telescope.lenses`,
and :mod:`iactrace.telescope.obstructions`. The key acceptance criteria are
that each helper returns the right domain type with the right shapes /
values, and that the resulting objects can be handed straight to
:class:`Telescope` and round-tripped through the render pipeline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from iactrace.core.apertures import DiskAperture, PolygonAperture
from iactrace.core.bsdf import DoubleGaussianBSDF, GaussianBSDF
from iactrace.core.interactions import (
    InteractionType,
    RefractInteraction,
    SlabInteraction,
)
from iactrace.core.obstructions import (
    BoxGroup,
    CylinderGroup,
    OpenCylinderGroup,
    SphereGroup,
)
from iactrace.telescope import Telescope, lenses, mirrors, obstructions

# mirrors.*


class TestMirrorsSpherical:
    def test_conic_and_curvature(self, random_key):
        m = mirrors.spherical(
            position=(0.0, 0.0, 0.0),
            focal_length=0.4,
            radius=0.1,
            key=random_key,
        )
        # c = 1 / (2 * f) = 1.25
        assert jnp.allclose(m.surface.curvatures, jnp.array([1.25]))
        assert jnp.allclose(m.surface.conics, jnp.array([0.0]))
        assert m.interaction == InteractionType.REFLECT

    def test_bsdf_set_when_nonzero(self, random_key):
        m = mirrors.spherical(
            position=(0, 0, 0),
            focal_length=0.4,
            radius=0.1,
            bsdf_scale=50.0,
            key=random_key,
        )
        assert isinstance(m.bsdf, GaussianBSDF)
        assert jnp.allclose(m.bsdf.scale, jnp.array([50.0]))


class TestMirrorsParabolic:
    def test_matches_cassegrain_yaml_primary(self, random_key):
        # configs/BASIC/Cassegrain_telescope.yaml primary: curvature=1.25, conic=-1
        m = mirrors.parabolic(
            position=(0.0, 0.0, 0.0),
            focal_length=0.4,
            radius=0.1,
            inner_radius=0.03,
            key=random_key,
        )
        assert jnp.allclose(m.surface.curvatures, jnp.array([1.25]))
        assert jnp.allclose(m.surface.conics, jnp.array([-1.0]))
        assert jnp.allclose(m.aperture.radii, jnp.array([0.1]))
        assert jnp.allclose(m.aperture.inner_radii, jnp.array([0.03]))


class TestMirrorsAspheric:
    def test_matches_cassegrain_yaml_secondary(self, random_key):
        # Secondary: curvature=-4.444444, conic=-2.25, rotation=(180,0,0)
        m = mirrors.aspheric(
            position=(0.0, 0.0, 0.31),
            rotation=(180.0, 0.0, 0.0),
            curvature=-4.444444,
            conic=-2.25,
            radius=0.032,
            optical_stage=1,
            key=random_key,
        )
        assert jnp.allclose(m.surface.curvatures, jnp.array([-4.444444]), atol=1e-5)
        assert jnp.allclose(m.surface.conics, jnp.array([-2.25]))
        assert m.optical_stage == 1
        assert jnp.allclose(m.rotations[0], jnp.array([180.0, 0.0, 0.0]))

    def test_aspheric_coeffs(self, random_key):
        m = mirrors.aspheric(
            position=(0, 0, 0),
            curvature=1.0,
            radius=0.1,
            aspheric_coeffs=(1e-4, 2e-6),
            key=random_key,
        )
        assert m.surface.aspherics.shape == (1, 2)
        assert jnp.allclose(m.surface.aspherics[0], jnp.array([1e-4, 2e-6]))


class TestMirrorsDiskArray:
    def test_batches_n_elements(self, random_key):
        n = 4
        m = mirrors.disk_array(
            positions=[[i * 0.1, 0.0, 0.0] for i in range(n)],
            rotations=[[0.0, 0.0, 0.0]] * n,
            curvatures=[1.25] * n,
            radii=[0.05] * n,
            conics=[-1.0] * n,
            key=random_key,
        )
        assert len(m) == n
        assert m.positions.shape == (n, 3)
        assert m.surface.curvatures.shape == (n,)
        assert m.aperture.radii.shape == (n,)
        assert jnp.allclose(m.surface.conics, jnp.full(n, -1.0))
        assert m.interaction == InteractionType.REFLECT

    def test_defaults_fill_in(self, random_key):
        m = mirrors.disk_array(
            positions=[[0, 0, 0], [0.1, 0, 0]],
            rotations=[[0, 0, 0], [0, 0, 0]],
            curvatures=[1.0, 2.0],
            radii=[0.05, 0.05],
            key=random_key,
        )
        assert jnp.allclose(m.surface.conics, jnp.zeros(2))
        assert jnp.allclose(m.aperture.inner_radii, jnp.zeros(2))
        assert jnp.allclose(m.interaction_module.reflectivity_scalar, jnp.ones(2))

    def test_shape_mismatch_raises(self, random_key):
        with pytest.raises(ValueError, match="curvatures"):
            mirrors.disk_array(
                positions=[[0, 0, 0], [0.1, 0, 0]],
                rotations=[[0, 0, 0], [0, 0, 0]],
                curvatures=[1.0],  # wrong length
                radii=[0.05, 0.05],
                key=random_key,
            )

    def test_bsdf_enabled_with_any_nonzero(self, random_key):
        m = mirrors.disk_array(
            positions=[[0, 0, 0], [0.1, 0, 0]],
            rotations=[[0, 0, 0], [0, 0, 0]],
            curvatures=[1.0, 1.0],
            radii=[0.05, 0.05],
            bsdf_scales=[0.0, 30.0],
            key=random_key,
        )
        assert isinstance(m.bsdf, GaussianBSDF)
        assert jnp.allclose(m.bsdf.scale, jnp.array([0.0, 30.0]))


# lenses.*


class TestLensesAsphericLens:
    def test_explicit_curvature_and_conic(self, random_key):
        lens = lenses.aspheric_lens(
            position=(0, 0, 0.1),
            curvature=7.5,
            conic=-0.5,
            aspheric_coeffs=(1e-3,),
            radius=0.02,
            key=random_key,
        )
        assert jnp.allclose(lens.surface.curvatures, jnp.array([7.5]))
        assert jnp.allclose(lens.surface.conics, jnp.array([-0.5]))
        assert lens.surface.aspherics.shape == (1, 1)
        assert isinstance(lens.interaction_module, RefractInteraction)


class TestLensesPlanoSlab:
    def test_thickness_passed_through(self, random_key):
        slab = lenses.plano_slab(
            position=(0, 0, 0.39),
            radius=0.05,
            thickness=0.003,
            n_inside=1.52,
            transmittance=0.9,
            key=random_key,
        )
        assert jnp.allclose(slab.interaction_module.thickness, jnp.array([0.003]))
        assert jnp.allclose(slab.interaction_module.n_inside, jnp.array([1.52]))
        assert jnp.allclose(slab.interaction_module.transmittance_scalar, jnp.array([0.9]))
        assert slab.interaction == InteractionType.SLAB


# obstructions.*


class TestObstructionsPrimitives:
    def test_cylinder(self):
        c = obstructions.cylinder(p1=(0, 0, 0.315), p2=(0, 0, 0.330), r=0.032)
        assert isinstance(c, CylinderGroup)
        assert len(c) == 1
        assert jnp.allclose(c.p1[0], jnp.array([0.0, 0.0, 0.315]))
        assert jnp.allclose(c.p2[0], jnp.array([0.0, 0.0, 0.330]))
        assert jnp.allclose(c.r, jnp.array([0.032]))

    def test_open_cylinder(self):
        c = obstructions.open_cylinder(p1=(0, 0, 0), p2=(0, 0, 1), r=0.01)
        assert isinstance(c, OpenCylinderGroup)
        assert len(c) == 1
        assert jnp.allclose(c.r, jnp.array([0.01]))

    def test_box(self):
        b = obstructions.box(p1=(-0.1, -0.1, 0.0), p2=(0.1, 0.1, 0.05))
        assert isinstance(b, BoxGroup)
        assert len(b) == 1
        assert jnp.allclose(b.p1[0], jnp.array([-0.1, -0.1, 0.0]))

    def test_sphere(self):
        s = obstructions.sphere(center=(0, 0, 0.4), r=0.02)
        assert isinstance(s, SphereGroup)
        assert len(s) == 1
        assert jnp.allclose(s.radii, jnp.array([0.02]))


# End-to-end: build a Telescope from helpers and render


class TestTelescopeEndToEnd:
    def test_build_parabolic_from_helpers_and_render(self):
        """Rebuild configs/BASIC/Parabolic_telescope.yaml via helpers and render.

        We only check that the pipeline runs and produces rays; not that
        the image matches the YAML render pixel-for-pixel. A tight match
        would duplicate the yaml tests and is unnecessary: the helpers
        delegate to the same core primitives.
        """
        k_mirror, k_src = jax.random.split(jax.random.key(0))

        primary = mirrors.parabolic(
            position=(0.0, 0.0, 0.0),
            focal_length=0.4,
            radius=0.1,
            n_samples=256,
            key=k_mirror,
        )
        housing = obstructions.cylinder(
            p1=(0.0, 0.0, 0.401),
            p2=(0.0, 0.0, 0.420),
            r=0.03,
        )

        tel = Telescope(
            mirror_groups=[primary],
            obstruction_groups=[housing],
            camera_position=jnp.array([0.0, 0.0, 0.4]),
        )

        # On-axis parallel source
        directions = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])
        rb = tel.render(directions, values, source_type="parallel").materialise()

        # Rays should have been generated and transformed; total flux > 0.
        assert rb.values.shape[0] > 0
        assert float(jnp.sum(rb.values)) > 0.0

    def test_telescope_with_lens_group(self):
        """Mirror + slab lens at different optical stages."""
        k1, k2 = jax.random.split(jax.random.key(1))

        primary = mirrors.parabolic(
            position=(0, 0, 0),
            focal_length=0.4,
            radius=0.1,
            optical_stage=0,
            n_samples=64,
            key=k1,
        )
        window = lenses.plano_slab(
            position=(0, 0, 0.39),
            radius=0.05,
            thickness=0.002,
            n_inside=1.5,
            optical_stage=1,
            n_samples=64,
            key=k2,
        )

        tel = Telescope(
            mirror_groups=[primary],
            lens_groups=[window],
            camera_position=jnp.array([0.0, 0.0, 0.4]),
        )

        # Two groups at distinct stages should be okay.
        assert len(tel.optical_groups) == 2

        # Full render smoke test.
        directions = jnp.array([[0.0, 0.0, -1.0]])
        values = jnp.array([1.0])
        rb = tel.render(directions, values, source_type="parallel").materialise()
        assert rb.values.shape[0] > 0


# Low-level canonical builders: mirror_group / refractive_group / slab_group


class TestMirrorGroup:
    def test_accepts_nonzero_offsets(self, random_key):
        """mirror_group exposes per-element surface offsets that the
        sugar helpers default to zero."""
        offsets = jnp.array([[0.01, -0.02], [0.0, 0.0]])
        m = mirrors.mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            rotations=jnp.zeros((2, 3)),
            curvatures=jnp.array([1.25, 1.25]),
            conics=jnp.array([-1.0, -1.0]),
            aspherics=jnp.zeros((2, 0)),
            offsets=offsets,
            aperture=DiskAperture(
                radii=jnp.array([0.1, 0.1]),
                inner_radii=jnp.zeros(2),
            ),
            reflectivity=jnp.ones(2),
            sample_key=random_key,
        )
        assert jnp.allclose(m.surface.offsets, offsets)
        assert m.interaction == InteractionType.REFLECT

    def test_accepts_polygon_aperture(self, random_key):
        """mirror_group is the only entry-point that supports hex/polygon
        apertures"""
        # Equilateral hexagon with circumradius 0.1, centred on origin
        angles = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
        hex_verts = jnp.stack([0.1 * jnp.cos(angles), 0.1 * jnp.sin(angles)], axis=-1)
        aperture = PolygonAperture(
            vertices=hex_verts[None, :, :],
            n_vertices=6,
        )

        m = mirrors.mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([1.25]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=aperture,
            reflectivity=jnp.ones(1),
            sample_key=random_key,
        )
        assert isinstance(m.aperture, PolygonAperture)
        assert m.aperture.n_vertices == 6

    def test_accepts_double_gaussian_bsdf(self, random_key):
        """mirror_group takes a pre-built BSDF, so advanced roughness
        models (like DoubleGaussianBSDF) work without special helper
        support."""
        bsdf = DoubleGaussianBSDF(
            scale_narrow=jnp.array([10.0]),
            scale_wide=jnp.array([100.0]),
            mix_weight=jnp.array([0.2]),
        )
        m = mirrors.mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([1.25]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.1]),
                inner_radii=jnp.zeros(1),
            ),
            reflectivity=jnp.ones(1),
            bsdf=bsdf,
            sample_key=random_key,
        )
        assert isinstance(m.bsdf, DoubleGaussianBSDF)
        assert jnp.allclose(m.bsdf.scale_narrow, jnp.array([10.0]))
        assert jnp.allclose(m.bsdf.scale_wide, jnp.array([100.0]))

    def test_sugar_helpers_route_through_mirror_group(self, random_key):
        """The sugar helpers must produce groups that are structurally
        indistinguishable from a direct mirror_group call with the same
        inputs."""
        sugar = mirrors.parabolic(
            position=(0.0, 0.0, 0.0),
            focal_length=0.4,
            radius=0.1,
            key=random_key,
        )
        direct = mirrors.mirror_group(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            rotations=jnp.zeros((1, 3)),
            curvatures=jnp.array([1.25]),
            conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)),
            offsets=jnp.zeros((1, 2)),
            aperture=DiskAperture(
                radii=jnp.array([0.1]),
                inner_radii=jnp.zeros(1),
            ),
            reflectivity=jnp.ones(1),
            sample_key=random_key,
        )
        # Same surface
        assert jnp.allclose(sugar.surface.curvatures, direct.surface.curvatures)
        assert jnp.allclose(sugar.surface.conics, direct.surface.conics)
        assert jnp.allclose(sugar.surface.offsets, direct.surface.offsets)
        # Same aperture
        assert jnp.allclose(sugar.aperture.radii, direct.aperture.radii)
        # Same interaction
        assert jnp.allclose(
            sugar.interaction_module.reflectivity_scalar,
            direct.interaction_module.reflectivity_scalar,
        )


class TestRefractiveGroup:
    def test_builds_group_with_refract_interaction(self, random_key):
        lens = lenses.refractive_group(
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
            transmittance=jnp.array([1.0]),
            sample_key=random_key,
        )
        assert isinstance(lens.interaction_module, RefractInteraction)
        assert lens.interaction == InteractionType.REFRACT
        assert jnp.allclose(lens.interaction_module.n_inside, jnp.array([1.5]))
        assert lens.interaction_module.n_outside == 1.0


class TestSlabGroup:
    def test_slab_group_hardcodes_flat_surface(self, random_key):
        slab = lenses.slab_group(
            positions=jnp.array([[0.0, 0.0, 0.39]]),
            rotations=jnp.zeros((1, 3)),
            aperture=DiskAperture(
                radii=jnp.array([0.05]),
                inner_radii=jnp.zeros(1),
            ),
            n_inside=jnp.array([1.5]),
            n_outside=1.0,
            thickness=jnp.array([0.002]),
            transmittance=jnp.array([1.0]),
            sample_key=random_key,
        )
        assert isinstance(slab.interaction_module, SlabInteraction)
        assert jnp.allclose(slab.surface.curvatures, jnp.zeros(1))
        assert jnp.allclose(slab.surface.conics, jnp.zeros(1))
        assert slab.surface.aspherics.shape == (1, 0)
        assert jnp.allclose(slab.interaction_module.thickness, jnp.array([0.002]))
