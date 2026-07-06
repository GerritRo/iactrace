"""YAML round-trip tests for Zernike figure-error surfaces."""

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


def _disk_config(mirrors):
    return {
        "telescope": {
            "name": "z",
            "units": "m",
            "camera_position": [0.0, 0.0, 10.0],
            "camera_rotation": [0.0, 0.0, 0.0],
        },
        "mirror_templates": {
            "sph": {"surface": {"curvature": 0.05, "conic": -1.0, "aspheric": []}}
        },
        "mirrors": mirrors,
        "obstructions": [],
    }


def _mirror(id_, pos, zernike=None):
    m = {
        "id": id_,
        "template": "sph",
        "position": pos,
        "orientation": [0.0, 0.0, 0.0],
        "aperture": {"type": "circular", "radius": 0.5},
    }
    if zernike is not None:
        m["zernike"] = zernike
    return m


KEY = jax.random.key(0)


class TestZernikeLoad:
    def test_loads_as_sum_surface(self):
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0, 0.0, 0.0, 1e-3, 5e-4], "r_norm": 0.5}),
        ])
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3, 5e-4])
        assert float(zern.r_norm[0]) == pytest.approx(0.5)

    def test_no_zernike_stays_bare_asphere(self):
        cfg = _disk_config([_mirror("M_0", [0.0, 0.0, 0.0])])
        tel = build_telescope_config(cfg, 4, KEY)
        assert isinstance(tel.stage(0).surface, AsphericSurfaceGroup)

    def test_mixed_bucket_zero_fills_absent(self):
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0, 0.0, 0.0, 1e-3], "r_norm": 0.5}),
            _mirror("M_1", [1.0, 0.0, 0.0]),  # no zernike
        ])
        tel = build_telescope_config(cfg, 4, KEY)
        surface = tel.stage(0).surface
        assert isinstance(surface, SumSurfaceGroup)
        zern = next(c for c in surface.components if isinstance(c, ZernikeSurfaceGroup))
        # element 0 has the defocus term, element 1 is all zero
        assert np.allclose(np.asarray(zern.coeffs[0]), [0.0, 0.0, 0.0, 1e-3])
        assert np.allclose(np.asarray(zern.coeffs[1]), 0.0)


class TestZernikeRoundTrip:
    def test_idempotent_dict_round_trip(self):
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4], "r_norm": 0.5}),
            _mirror("M_1", [1.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0, 0.0, 0.0, 2e-3], "r_norm": 0.5}),
        ])
        tel = build_telescope_config(cfg, 4, KEY)
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        assert any("zernike" in m for m in d1["mirrors"])

    def test_sag_preserved(self):
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0, 0.0, 0.0, 1e-3, 5e-4, -5e-4], "r_norm": 0.5}),
        ])
        tel = build_telescope_config(cfg, 4, KEY)
        tel2 = build_telescope_config(telescope_to_dict(tel), 4, KEY)
        s1, s2 = tel.stage(0).surface, tel2.stage(0).surface
        for x, y in [(0.1, 0.05), (-0.2, 0.1), (0.3, -0.25)]:
            assert float(s1.sag_at(0, x, y)) == pytest.approx(
                float(s2.sag_at(0, x, y)), abs=1e-9
            )

    def test_absent_zernike_not_emitted(self):
        cfg = _disk_config([_mirror("M_0", [0.0, 0.0, 0.0])])
        tel = build_telescope_config(cfg, 4, KEY)
        d = telescope_to_dict(tel)
        assert all("zernike" not in m for m in d["mirrors"])

    def test_perturbed_telescope_round_trips(self):
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0]),
            _mirror("M_1", [1.0, 0.0, 0.0]),
        ])
        tel = build_telescope_config(cfg, 4, KEY)
        tel = ops.apply_astigmatism(tel, 0, 1e-3, jax.random.key(5))
        d1 = telescope_to_dict(tel)
        tel2 = build_telescope_config(d1, 4, KEY)
        d2 = telescope_to_dict(tel2)
        assert d1 == d2
        # every facet got a draw -> every mirror carries a zernike block
        assert all("zernike" in m for m in d1["mirrors"])


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
    def test_standalone_zernike_serializes_as_flat_plus_zernike(self):
        zg = ZernikeSurfaceGroup(
            coeffs=jnp.array([[0.0, 0.0, 0.0, 1e-3, 5e-4]]), r_norm=jnp.array([0.5]),
        )
        tel = Telescope(mirror_groups=[_mirror_group(zg, jnp.array([0.5]))], name="z")
        d = telescope_to_dict(tel)
        # flat aspheric base + a zernike block
        assert d["mirrors"][0]["zernike"]["coeffs"][3] == pytest.approx(1e-3)
        tname = d["mirrors"][0]["template"]
        assert d["mirror_templates"][tname]["surface"]["curvature"] == 0.0
        # reloads to an equivalent surface
        tel2 = build_telescope_config(d, 4, KEY)
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
        cfg = _disk_config([
            _mirror("M_0", [0.0, 0.0, 0.0],
                    zernike={"coeffs": [0.0] * 12, "r_norm": 0.5}),
        ])
        with pytest.raises(YAMLConfigError):
            build_telescope_config(cfg, 4, KEY)
