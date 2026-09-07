"""Saving an effective-aperture table, and reading it back.

The scan is the expensive step, so the file is what consumers actually use.
Two things have to hold: a table survives the round trip exactly, and the
file stays readable with numpy alone -- no iactrace, no JAX -- because that
is what lets ``nyx`` read it.
"""

import json
import zipfile

import numpy as np
import pytest

from iactrace.analysis import EffectiveApertureTable, effective_aperture
from iactrace.io import (
    FORMAT,
    FORMAT_VERSION,
    load_aperture_table,
    read_aperture_table_arrays,
    save_aperture_table,
)

from .test_effective_aperture import BAND, HALF_ANGLE, STEP, make_rig

ARRAY_FIELDS = ("origin", "step", "offset", "values", "wavelengths", "spectral_area")


@pytest.fixture(scope="module")
def table():
    """One scanned table, shared: the scan is far slower than any assertion."""
    return effective_aperture(
        *make_rig(spectral=True), half_angle=HALF_ANGLE, step=STEP, wavelengths=BAND
    )


def assert_same_table(a, b):
    for name in ARRAY_FIELDS:
        lhs, rhs = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert lhs.shape == rhs.shape, name
        assert np.array_equal(lhs.astype(rhs.dtype), rhs), name
    assert a.on_axis_area == b.on_axis_area


class TestRoundTrip:
    def test_arrays_survive_exactly(self, table, tmp_path):
        """Bit-for-bit, not approximately: nothing here is recomputed on load."""
        path = save_aperture_table(table, tmp_path / "rig.npz")
        assert_same_table(table, load_aperture_table(path))

    def test_derived_quantities_agree(self, table, tmp_path):
        """The properties a consumer reads, not just the fields stored."""
        back = load_aperture_table(save_aperture_table(table, tmp_path / "rig.npz"))
        assert back.n_pixels == table.n_pixels
        assert back.window == table.window
        assert np.allclose(back.centres, table.centres)
        assert np.allclose(back.window_centres, table.window_centres)

    def test_meta_survives_as_json(self, table, tmp_path):
        """Provenance carries numpy scalars and tuples; JSON has neither."""
        back = load_aperture_table(save_aperture_table(table, tmp_path / "rig.npz"))
        assert back.meta["telescope"] == table.meta["telescope"]
        assert back.meta["n_samples"] == int(table.meta["n_samples"])
        # A tuple has no JSON counterpart and comes back as a list.
        assert list(back.meta["pixel_shape"]) == list(table.meta["pixel_shape"])
        assert isinstance(json.dumps(back.meta), str)

    def test_methods_are_the_same_round_trip(self, table, tmp_path):
        """``table.save`` / ``EffectiveApertureTable.load`` are the same path."""
        path = table.save(tmp_path / "rig.npz")
        assert_same_table(table, EffectiveApertureTable.load(path))

    def test_uncompressed_is_equivalent(self, table, tmp_path):
        compressed = save_aperture_table(table, tmp_path / "small.npz")
        plain = save_aperture_table(table, tmp_path / "plain.npz", compress=False)
        assert_same_table(load_aperture_table(compressed), load_aperture_table(plain))
        # A response cube is mostly zeros, so deflating it should actually pay.
        assert compressed.stat().st_size < plain.stat().st_size


class TestFile:
    def test_suffix_is_added(self, table, tmp_path):
        assert save_aperture_table(table, tmp_path / "rig").name == "rig.npz"

    def test_parent_directory_is_created(self, table, tmp_path):
        path = save_aperture_table(table, tmp_path / "nested" / "deeper" / "rig.npz")
        assert path.is_file()

    def test_overwrite_can_be_refused(self, table, tmp_path):
        path = save_aperture_table(table, tmp_path / "rig.npz")
        with pytest.raises(FileExistsError):
            save_aperture_table(table, path, overwrite=False)
        save_aperture_table(table, path, overwrite=True)  # the default

    def test_archive_is_self_describing(self, table, tmp_path):
        """A reader that has never seen iactrace can still tell what it holds."""
        path = save_aperture_table(table, tmp_path / "rig.npz")
        with zipfile.ZipFile(path) as archive:
            keys = {name.removesuffix(".npy") for name in archive.namelist()}
        assert keys == {*ARRAY_FIELDS, "on_axis_area", "meta", "format", "format_version"}


class TestNumpyOnlyReader:
    """The half a consumer without iactrace uses."""

    def test_arrays_read_without_building_a_table(self, table, tmp_path):
        path = save_aperture_table(table, tmp_path / "rig.npz")
        fields = read_aperture_table_arrays(path)
        assert set(fields) == {*ARRAY_FIELDS, "on_axis_area", "meta"}
        assert_same_table(table, EffectiveApertureTable(**fields))

    def test_nothing_in_the_file_is_pickled(self, table, tmp_path):
        """Loading a table must never execute code from it."""
        path = save_aperture_table(table, tmp_path / "rig.npz")
        with np.load(path, allow_pickle=False) as archive:  # would raise if it were
            assert str(archive["format"]) == FORMAT
            assert str(archive["format_version"]) == FORMAT_VERSION


class TestRejection:
    def test_a_foreign_npz_is_named_as_such(self, tmp_path):
        path = tmp_path / "junk.npz"
        np.savez(path, values=np.zeros(3))
        with pytest.raises(ValueError, match=FORMAT):
            load_aperture_table(path)

    def test_a_future_major_version_is_refused(self, table, tmp_path):
        """Better to stop than to read a layout that has since changed."""
        path = save_aperture_table(table, tmp_path / "rig.npz")
        with np.load(path, allow_pickle=False) as archive:
            payload = dict(archive.items())
        payload["format_version"] = np.asarray("99.0")
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="99.0"):
            load_aperture_table(path)

    def test_a_truncated_file_names_what_is_missing(self, table, tmp_path):
        path = save_aperture_table(table, tmp_path / "rig.npz")
        with np.load(path, allow_pickle=False) as archive:
            payload = {k: v for k, v in archive.items() if k != "spectral_area"}
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="spectral_area"):
            load_aperture_table(path)
