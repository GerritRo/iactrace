"""YAML round-trip tests for Zernike figure surfaces (as surface-list terms)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iactrace import Telescope
from iactrace.core.apertures import DiskAperture
from iactrace.core.interactions import ReflectInteraction
from iactrace.core.optics import OpticalElementGroup
from iactrace.core.surfaces import (
    AsphericSurfaceGroup,
    SumSurfaceGroup,
    ZernikeSurfaceGroup,
)
from iactrace.io import build_telescope_config, telescope_to_dict
from iactrace.io.yaml_io import YAMLConfigError
from iactrace.telescope import operations as ops

KEY = jax.random.key(0)


def _config(templates, mirrors):
    return {
        "telescope": {
            "name": "z",
            "units": "m",
            "camera_position": [0.0, 0.0, 10.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": templates,
        "mirrors": mirrors,
        "obstructions": [],
    }


def _mirror(id_, pos, template="sph"):
    return {
        "id": id_,
        "template": template,
        "position": pos,
        "orientation": [0.0, 0.0, 0.0],
        "aperture": {"type": "circular", "radius": 0.5},
    }


def _asphere(curvature=0.05, conic=-1.0):
    return {"type": "aspheric", "curvature": curvature, "conic": conic, "aspheric": []}


def _zernike(coeffs, r_norm=0.5):
    return {"type": "zernike", "coeffs": coeffs, "r_norm": r_norm}


def _has_zernike(surface_spec):
    shapes = surface_spec if isinstance(surface_spec, list) else [surface_spec]
    return any(s.get("type") == "zernike" for s in shapes)


class TestSurfaceListLoad:
    def test_aspheric_plus_zernike_loads_as_sum(self):
        cfg = _config(
            {"sph": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4])]}},
            [_mirror("M_0", [0.0, 0.0, 0.0])],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3, 5e-4])
        assert float(zern.r_norm[0]) == pytest.approx(0.5)

    def test_single_aspheric_stays_bare_asphere(self):
        cfg = _config({"sph": {"surface": _asphere()}}, [_mirror("M_0", [0.0, 0.0, 0.0])])
        tel = build_telescope_config(cfg, 4, KEY)
        assert isinstance(tel.stage(0).surface, AsphericSurfaceGroup)

    def test_standalone_zernike_loads_as_zernike(self):
        cfg = _config(
            {"z": {"surface": _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4])}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="z")],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        assert isinstance(tel.stage(0).surface, ZernikeSurfaceGroup)

    def test_mixed_templates_zero_fill_absent(self):
        # two templates in the same stage+aperture bucket: only one has a zernike
        cfg = _config(
            {
                "zt": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3])]},
                "at": {"surface": _asphere()},
            },
            [
                _mirror("M_0", [0.0, 0.0, 0.0], template="zt"),
                _mirror("M_1", [1.0, 0.0, 0.0], template="at"),
            ],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3])
        assert np.allclose(np.asarray(zern.coeffs[1]), 0.0)

    def test_shared_template_shares_figure(self):
        cfg = _config(
            {"sph": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3])]}},
            [_mirror("M_0", [0.0, 0.0, 0.0]), _mirror("M_1", [1.0, 0.0, 0.0])],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        zern = next(c for c in tel.stage(0).surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), np.asarray(zern.coeffs[1]))


class TestRoundTrip:
    def test_idempotent_dict_round_trip(self):
        cfg = _config(
            {
                "a": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4])]},
                "b": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 2e-3])]},
            },
            [
                _mirror("M_0", [0.0, 0.0, 0.0], template="a"),
                _mirror("M_1", [1.0, 0.0, 0.0], template="b"),
            ],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        assert any(_has_zernike(t["surface"]) for t in d1["mirror_templates"].values())

    def test_sag_preserved(self):
        cfg = _config(
            {"a": {"surface": [_asphere(), _zernike([0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4])]}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="a")],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        tel2 = build_telescope_config(telescope_to_dict(tel), 4, KEY)
        s1, s2 = tel.stage(0).surface, tel2.stage(0).surface
        for x, y in [(0.1, 0.05), (-0.2, 0.1), (0.3, -0.25)]:
            assert float(s1.sag_at(0, x, y)) == pytest.approx(float(s2.sag_at(0, x, y)), abs=1e-9)

    def test_bare_asphere_has_no_zernike(self):
        cfg = _config({"sph": {"surface": _asphere()}}, [_mirror("M_0", [0.0, 0.0, 0.0])])
        tel = build_telescope_config(cfg, 4, KEY)
        d = telescope_to_dict(tel)
        assert not any(_has_zernike(t["surface"]) for t in d["mirror_templates"].values())

    def test_perturbed_telescope_round_trips(self):
        cfg = _config(
            {"sph": {"surface": _asphere()}},
            [_mirror("M_0", [0.0, 0.0, 0.0]), _mirror("M_1", [1.0, 0.0, 0.0])],
        )
        tel = build_telescope_config(cfg, 4, KEY)
        tel = ops.apply_astigmatism(tel, 0, 1e-3, jax.random.key(5))
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        # every facet drew a figure error -> some template carries a zernike shape
        assert any(_has_zernike(t["surface"]) for t in d1["mirror_templates"].values())


def _mirror_group(surface, radii, stage=0):
    n = surface.offsets.shape[0]
    aperture = DiskAperture(radii=radii, inner_radii=jnp.zeros(n))
    interaction = ReflectInteraction(reflectivity=None, reflectivity_scalar=jnp.ones(n))
    return OpticalElementGroup(
        positions=jnp.zeros((n, 3)),
        rotations=jnp.zeros((n, 3)),
        surface=surface,
        aperture=aperture,
        interaction_module=interaction,
        sample_key=jax.random.key(0),
        optical_stage=stage,
        n_samples=8,
    )


class TestStandaloneAndGuards:
    def test_standalone_zernike_serializes_as_zernike_surface(self):
        zg = ZernikeSurfaceGroup(
            coeffs=jnp.array([[0.0, 0.0, 0.0, 1e-3, 5e-4]]), r_norm=jnp.array([0.5]),
        )
        tel = Telescope(mirror_groups=[_mirror_group(zg, jnp.array([0.5]))], name="z")
        d = telescope_to_dict(tel)
        tname = d["mirrors"][0]["template"]
        surf = d["mirror_templates"][tname]["surface"]
        # a single standalone zernike shape (no aspheric base)
        assert surf["type"] == "zernike"
        assert surf["coeffs"][3] == pytest.approx(1e-3)
        # reloads to a standalone Zernike surface with an equivalent sag
        tel2 = build_telescope_config(d, 4, KEY)
        assert isinstance(tel2.stage(0).surface, ZernikeSurfaceGroup)
        for x, y in [(0.1, 0.05), (-0.2, 0.1)]:
            assert float(tel.stage(0).surface.sag_at(0, x, y)) == pytest.approx(
                float(tel2.stage(0).surface.sag_at(0, x, y)), abs=1e-9
            )

    def test_nonzero_composite_offset_raises(self):
        asph = AsphericSurfaceGroup(
            curvatures=jnp.array([0.05]), conics=jnp.array([-1.0]),
            aspherics=jnp.zeros((1, 0)), offsets=jnp.zeros((1, 2)),
        )
        zg = ZernikeSurfaceGroup(coeffs=jnp.zeros((1, 4)), r_norm=jnp.array([0.5]))
        bad = SumSurfaceGroup([asph, zg], offsets=jnp.array([[0.1, 0.0]]))
        tel = Telescope(mirror_groups=[_mirror_group(bad, jnp.array([0.5]))], name="z")
        with pytest.raises(ValueError, match="composite decenter"):
            telescope_to_dict(tel)

    def test_zernike_too_many_coeffs_rejected_by_schema(self):
        cfg = _config(
            {"z": {"surface": _zernike([0.0] * 12)}},
            [_mirror("M_0", [0.0, 0.0, 0.0], template="z")],
        )
        with pytest.raises(YAMLConfigError):
            build_telescope_config(cfg, 4, KEY)