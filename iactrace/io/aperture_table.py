from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "load_aperture_table",
    "read_aperture_table_arrays",
    "save_aperture_table",
]

logger = logging.getLogger(__name__)

#: Marker identifying an effective-aperture archive
FORMAT = "iactrace-aperture-table"

#: Format version, major.minor.
FORMAT_VERSION = "1.0"

#: Arrays that make up a table, in the order the dataclass declares them.
_ARRAYS = ("origin", "step", "offset", "values", "wavelengths", "spectral_area")

_DTYPES = {
    "origin": np.float64,
    "step": np.float64,
    "offset": np.int32,
    "values": np.float32,
    "wavelengths": np.float64,
    "spectral_area": np.float64,
}


def save_aperture_table(
    table: Any,
    filename: str | Path,
    *,
    overwrite: bool = True,
    compress: bool = True,
) -> Path:
    """Write an effective-aperture table to a .npz file.

    Parameters
    ----------
    table
        The EffectiveApertureTable to write.
    filename
        Output path.
    overwrite
        If False, refuse to replace an existing file.
    compress
        Store deflated.

    Returns
    -------
    The path written.

    Raises
    ------
    FileExistsError
        if the file exists and overwrite is False.
    """
    filepath = Path(filename)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(filepath.suffix + ".npz")
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        name: np.asarray(getattr(table, name), dtype=_DTYPES[name]) for name in _ARRAYS
    }
    payload["on_axis_area"] = np.asarray(float(table.on_axis_area))
    payload["meta"] = np.asarray(json.dumps(_jsonable(table.meta)))
    payload["format"] = np.asarray(FORMAT)
    payload["format_version"] = np.asarray(FORMAT_VERSION)

    writer = np.savez_compressed if compress else np.savez
    with open(filepath, "wb") as f:
        writer(f, **payload)

    logger.info("Saved effective-aperture table to %s", filepath)
    return filepath


def load_aperture_table(filename: str | Path) -> Any:
    """Read an effective-aperture table written by save_aperture_table.

    Parameters
    ----------
    filename
        Path to the .npz archive.

    Returns
    -------
    The EffectiveApertureTable.

    Raises
    ------
    ValueError
        if the file is not an aperture table, or is a format
        version this reader does not know.
    """
    from ..analysis.effective_aperture import EffectiveApertureTable

    fields = read_aperture_table_arrays(filename)
    return EffectiveApertureTable(**fields)


def read_aperture_table_arrays(filename: str | Path) -> dict[str, Any]:
    """The table's fields as plain arrays, without constructing the table.

    Parameters
    ----------
    filename
        Path to the .npz archive.

    Returns
    -------
    {origin, step, offset, values, on_axis_area, wavelengths,
    spectral_area, meta}, ready to hand to the table's constructor.
    """
    filepath = Path(filename)
    with np.load(filepath, allow_pickle=False) as archive:
        _check_format(filepath, archive)
        fields: dict[str, Any] = {name: archive[name] for name in _ARRAYS}
        fields["on_axis_area"] = float(archive["on_axis_area"])
        fields["meta"] = json.loads(str(archive["meta"]))
    return fields


def _check_format(filepath: Path, archive: Any) -> None:
    """Reject a file that is not a table this reader understands."""
    if "format" not in archive or str(archive["format"]) != FORMAT:
        raise ValueError(
            f"{filepath} is not an {FORMAT} file; write one with iactrace.io.save_aperture_table"
        )
    version = str(archive["format_version"])
    if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
        raise ValueError(
            f"{filepath} is format {version}, and this iactrace reads "
            f"{FORMAT_VERSION.split('.')[0]}.x"
        )
    missing = [name for name in (*_ARRAYS, "on_axis_area", "meta") if name not in archive]
    if missing:
        raise ValueError(f"{filepath} is missing {', '.join(missing)}")


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars and containers into something JSON can hold."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value
