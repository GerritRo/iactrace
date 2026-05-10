from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core.ray_bundle import LazyRayBundle, RayBundle
from .focal_surface import FocalSurface, FocalSurfaceHits


def _materialise(rb: RayBundle | LazyRayBundle) -> RayBundle:
    if isinstance(rb, LazyRayBundle):
        return rb.materialise()
    return rb


def _hits_and_weights(
    surface: FocalSurface,
    ray_bundle: RayBundle | LazyRayBundle,
    weights: Array | None,
) -> tuple[FocalSurfaceHits, Array]:
    """Run the intersection and resolve the per-ray weight vector."""
    rb = _materialise(ray_bundle)
    hits = surface.intersect(rb)
    w = rb.values if weights is None else jnp.asarray(weights)
    # Zero out missed rays so downstream sums/histograms ignore them.
    w_eff = jnp.where(hits.hit_mask, w, 0.0)
    return hits, w_eff


def _weighted_centroid(xy: Array, w: Array) -> Array:
    """Weighted centroid (cx, cy). Returns zeros if total weight is zero."""
    total = jnp.sum(w)
    safe_total = jnp.where(total > 0, total, 1.0)
    cx = jnp.sum(w * xy[:, 0]) / safe_total
    cy = jnp.sum(w * xy[:, 1]) / safe_total
    return jnp.where(total > 0, jnp.stack([cx, cy]), jnp.zeros(2))


def _resolve_center(xy: Array, w: Array, center: Array | None) -> Array:
    if center is None:
        return _weighted_centroid(xy, w)
    return jnp.asarray(center, dtype=xy.dtype)


def spot_diagram(
    surface: FocalSurface,
    ray_bundle: RayBundle | LazyRayBundle,
    *,
    weights: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Per-ray hit positions on the focal surface.

    Args:
        surface: Focal surface to project onto.
        ray_bundle: Rays in the camera/telescope frame.
        weights: Optional per-ray weights; defaults to ``ray_bundle.values``.

    Returns:
        Tuple ``(xy, w, hit_mask)``:
            - ``xy`` (n_rays, 2): surface-local hit coordinates.
            - ``w`` (n_rays,): effective weights, zeroed for missed rays.
            - ``hit_mask`` (n_rays,): True for rays that hit the surface.
    """
    hits, w_eff = _hits_and_weights(surface, ray_bundle, weights)
    return hits.xy_local, w_eff, hits.hit_mask


def psf_image(
    surface: FocalSurface,
    ray_bundle: RayBundle | LazyRayBundle,
    *,
    bins: int | tuple[int, int] = 128,
    extent: tuple[tuple[float, float], tuple[float, float]] | Array | None = None,
    weights: Array | None = None,
    density: bool = False,
) -> tuple[Array, Array, Array]:
    """Binned 2D PSF on the focal surface.

    Args:
        surface: Focal surface to project onto.
        ray_bundle: Rays in the camera/telescope frame.
        bins: Bin count, scalar or ``(nx, ny)``.
        extent: ``((xmin, xmax), (ymin, ymax))`` in surface-local units.
            If ``None``, taken from the bounding box of the valid hits.
        weights: Optional per-ray weights; defaults to ``ray_bundle.values``.
        density: If True, divide by bin area and total weight.

    Returns:
        Tuple ``(image, x_edges, y_edges)``. Without weights, ``image.sum()``
        equals the number of valid hits.
    """
    hits, w_eff = _hits_and_weights(surface, ray_bundle, weights)
    x = hits.xy_local[:, 0]
    y = hits.xy_local[:, 1]

    if extent is None:
        # Use only valid hits to compute the bounding box; missed rays may
        # carry sentinel positions like 1e10 from intersect_plane.
        x_min = jnp.min(jnp.where(hits.hit_mask, x, jnp.inf))
        x_max = jnp.max(jnp.where(hits.hit_mask, x, -jnp.inf))
        y_min = jnp.min(jnp.where(hits.hit_mask, y, jnp.inf))
        y_max = jnp.max(jnp.where(hits.hit_mask, y, -jnp.inf))
        # Pad by one bin width to keep boundary hits inside the histogram.
        nx = bins if isinstance(bins, int) else bins[0]
        ny = bins if isinstance(bins, int) else bins[1]
        pad_x = (x_max - x_min) / jnp.maximum(nx - 1, 1)
        pad_y = (y_max - y_min) / jnp.maximum(ny - 1, 1)
        range_arg = jnp.array([[x_min, x_max + pad_x], [y_min, y_max + pad_y]])
    else:
        range_arg = jnp.asarray(extent, dtype=float)

    # Replace miss positions with values inside the range; their weight is
    # already zero, so they contribute nothing.
    x_safe = jnp.where(hits.hit_mask, x, range_arg[0, 0])
    y_safe = jnp.where(hits.hit_mask, y, range_arg[1, 0])

    image, x_edges, y_edges = jnp.histogram2d(
        x_safe, y_safe, bins=bins, range=range_arg,
        weights=w_eff, density=density,
    )
    return image, x_edges, y_edges


def rms_spot_size(
    surface: FocalSurface,
    ray_bundle: RayBundle | LazyRayBundle,
    *,
    weights: Array | None = None,
    center: Array | None = None,
) -> Array:
    """Weighted RMS radius of the spot on the focal surface.

    ``center`` defaults to the weighted centroid of the valid hits.

    Returns a scalar JAX array. Equal to
    ``sqrt(sum(w * |xy - center|^2) / sum(w))``.
    """
    hits, w_eff = _hits_and_weights(surface, ray_bundle, weights)
    c = _resolve_center(hits.xy_local, w_eff, center)

    dx = jnp.where(hits.hit_mask, hits.xy_local[:, 0] - c[0], 0.0)
    dy = jnp.where(hits.hit_mask, hits.xy_local[:, 1] - c[1], 0.0)
    r2 = dx * dx + dy * dy

    total = jnp.sum(w_eff)
    safe_total = jnp.where(total > 0, total, 1.0)
    rms = jnp.sqrt(jnp.sum(w_eff * r2) / safe_total)
    return jnp.where(total > 0, rms, 0.0)


def encircled_energy(
    surface: FocalSurface,
    ray_bundle: RayBundle | LazyRayBundle,
    *,
    n_radii: int = 200,
    r_max: float | Array | None = None,
    weights: Array | None = None,
    center: Array | None = None,
) -> tuple[Array, Array]:
    """Encircled-energy curve about ``center``.

    Args:
        surface: Focal surface to project onto.
        ray_bundle: Rays in the camera/telescope frame.
        n_radii: Number of sample points along the radius axis.
        r_max: Maximum radius. If ``None``, taken from the largest valid
            hit-to-center distance.
        weights: Optional per-ray weights; defaults to ``ray_bundle.values``.
        center: Centre of the circular apertures. Defaults to the weighted
            centroid.

    Returns:
        Tuple ``(radii, fraction)`` of length ``n_radii``. ``fraction[i]``
        is the cumulative fraction of total weight inside a circle of
        radius ``radii[i]``. ``fraction[-1]`` is 1.0 when ``r_max`` covers
        every valid hit.
    """
    hits, w_eff = _hits_and_weights(surface, ray_bundle, weights)
    c = _resolve_center(hits.xy_local, w_eff, center)

    dx = hits.xy_local[:, 0] - c[0]
    dy = hits.xy_local[:, 1] - c[1]
    r = jnp.sqrt(dx * dx + dy * dy)
    # Push missed rays beyond any reasonable radius so they never count.
    r_eff = jnp.where(hits.hit_mask, r, jnp.inf)

    if r_max is None:
        r_top = jnp.max(jnp.where(hits.hit_mask, r, 0.0))
    else:
        r_top = jnp.asarray(r_max, dtype=r.dtype)
    radii = jnp.linspace(0.0, r_top, n_radii)

    total = jnp.sum(w_eff)
    safe_total = jnp.where(total > 0, total, 1.0)

    # fraction(R) = sum_i w_i * 1[r_i <= R] / total. Vectorised over R; the
    # broadcast is (n_radii, n_rays) which is fine for analysis-scale data.
    inside = (r_eff[None, :] <= radii[:, None]).astype(w_eff.dtype)
    cumulative = jnp.sum(inside * w_eff[None, :], axis=1) / safe_total
    fraction = jnp.where(total > 0, cumulative, jnp.zeros_like(cumulative))
    return radii, fraction