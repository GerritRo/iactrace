from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from ...core.apertures import Aperture, DiskAperture, PolygonAperture
from ..schemas import (
    CircularApertureSchema,
    PolygonApertureSchema,
)


def _to_float_list(arr: np.ndarray | jnp.ndarray) -> list[float]:
    """Convert a JAX/NumPy array to a plain list of floats."""
    return [float(x) for x in np.asarray(arr)]


def _strip_trailing_zeros(values: list[float]) -> list[float]:
    """Strip trailing zero coefficients from an aspheric list."""
    while values and values[-1] == 0.0:
        values.pop()
    return values


def _pad_aspherics(aspheric_list: list[list[float]]) -> jnp.ndarray:
    """Pad aspheric coefficient arrays to uniform length.

    When all elements have empty coefficient lists, returns shape (N, 0)
    so that sag_raw skips the aspheric computation entirely.
    """
    if not aspheric_list:
        return jnp.zeros((0, 0))

    max_len = max(len(a) for a in aspheric_list)

    if max_len == 0:
        return jnp.zeros((len(aspheric_list), 0))

    padded = []
    for a in aspheric_list:
        arr = jnp.asarray(a)
        if len(arr) < max_len:
            arr = jnp.concatenate([arr, jnp.zeros(max_len - len(arr))])
        padded.append(arr)

    return jnp.stack(padded)


def _ensure_ccw(vertices: jnp.ndarray) -> jnp.ndarray:
    """Ensure polygon vertices are in counter-clockwise order."""
    vx, vy = vertices[:, 0], vertices[:, 1]
    signed_area = 0.5 * jnp.sum(vx * jnp.roll(vy, -1) - jnp.roll(vx, -1) * vy)
    return jnp.where(signed_area < 0, vertices[::-1], vertices)


def _disk_aperture_from_schemas(schemas: list[CircularApertureSchema]) -> DiskAperture:
    return DiskAperture(
        radii=jnp.asarray([s.radius for s in schemas]),
        inner_radii=jnp.asarray([s.inner_radius for s in schemas]),
    )


def _polygon_aperture_from_schemas(schemas: list[PolygonApertureSchema]) -> PolygonAperture:
    vertices = jnp.stack([_ensure_ccw(jnp.asarray(s.vertices)) for s in schemas])
    return PolygonAperture(vertices=vertices, n_vertices=int(vertices.shape[1]))


def _aperture_from_schemas(
    schemas: list[CircularApertureSchema | PolygonApertureSchema],
) -> Aperture:
    """Build a single Aperture from a list of homogeneous aperture schemas.

    All schemas must share the same kind (and, for polygons, the same
    vertex count); callers are expected to bucket via
    _bucket_by_aperture_signature first.
    """
    disks: list[CircularApertureSchema] = []
    polys: list[PolygonApertureSchema] = []
    for s in schemas:
        match s.type:
            case "circular":
                disks.append(s)
            case "polygon":
                polys.append(s)
    if disks and not polys:
        return _disk_aperture_from_schemas(disks)
    if polys and not disks:
        return _polygon_aperture_from_schemas(polys)
    raise ValueError("aperture schemas must be homogeneous (all disk or all polygon)")


def _bucket_by_aperture_signature[T](
    items: list[T],
    aperture_of: Callable[[T], CircularApertureSchema | PolygonApertureSchema],
) -> list[list[T]]:
    """Group items so each bucket has a single aperture signature.

    Disk apertures form one bucket; each distinct polygon vertex count
    forms its own bucket. Order within a bucket follows input order.
    """
    disk_bucket: list[T] = []
    poly_buckets: dict[int, list[T]] = defaultdict(list)
    for item in items:
        ap = aperture_of(item)
        match ap.type:
            case "polygon":
                poly_buckets[len(ap.vertices)].append(item)
            case "circular":
                disk_bucket.append(item)

    buckets: list[list[T]] = []
    if disk_bucket:
        buckets.append(disk_bucket)
    buckets.extend(poly_buckets.values())
    return buckets


def _aperture_to_schema(
    aperture: Aperture, i: int
) -> CircularApertureSchema | PolygonApertureSchema:
    """Convert a domain aperture element to its schema representation."""
    match aperture:
        case DiskAperture(radii=radii, inner_radii=inner_radii):
            return CircularApertureSchema(
                radius=float(radii[i]),
                inner_radius=float(inner_radii[i]),
            )
        case PolygonAperture(vertices=vertices):
            return PolygonApertureSchema(
                vertices=[[float(v[0]), float(v[1])] for v in np.asarray(vertices[i])],
            )
        case _:
            raise ValueError(f"Unknown aperture type: {type(aperture)}")
