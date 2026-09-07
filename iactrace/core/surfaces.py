from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .intersections import _localize, intersect_conic, newton_raphson_intersect
from .tolerances import dir_tol


def _default_offsets(n, offsets):
    """Resolve the optional per-element in-surface decenter to an (N, 2) array."""
    return jnp.zeros((n, 2)) if offsets is None else jnp.asarray(offsets)


# Surface group protocol


class SurfaceGroup(eqx.Module):
    """Abstract base for batched surface parameters of N optical elements.

    A SurfaceGroup stores per-element surface geometry and provides:
    - Sag/normal computation for the transform pipeline (vmapped per element)
    - Per-element sag and ray intersection for rendering and visualization
    """

    offsets: jax.Array  # (N, 2); per-element in-surface decenter

    @abstractmethod
    def _sag_intrinsic(self, x, y):
        """Intrinsic (centred, un-decentred) surface height z(x, y).

        Called on a single-element surface -- after the module has been sliced
        by _index or vmapped over axis 0 -- so per-element parameters
        are already scalar. The only method a new surface type must implement.
        """
        ...

    # Generic

    def _index(self, element_idx):
        """Slice the batched group down to a single element.

        Indexes every per-element array leaf along axis 0, leaving static
        fields untouched.
        """
        return jax.tree_util.tree_map(lambda a: a[element_idx], self)

    def _sag_local(self, x, y):
        """Intrinsic shape with the in-surface decenter applied and re-zeroed."""
        x0 = self.offsets[0]
        y0 = self.offsets[1]
        return self._sag_intrinsic(x + x0, y + y0) - self._sag_intrinsic(x0, y0)

    def _t_guess(self, ray_origin, ray_direction):
        """Initial ray parameter for the Newton intersection.

        Default: the tangent plane z = 0. Surfaces with a known analytic
        approximation override this for a better start.
        """
        dz = ray_direction[2]
        floor = dir_tol(dz)
        safe_dz = jnp.where(jnp.abs(dz) > floor, dz, floor)
        t = -ray_origin[2] / safe_dz
        return jnp.maximum(t, 0.0)

    @property
    def _t_guess_is_exact(self) -> bool:
        """Whether _t_guess already returns the exact intersection.

        True lets _intersect_t bypass the Newton iteration.
        """
        return False

    def _intersect_t(self, ray_origin, ray_direction):
        """Single-element nearest forward intersection parameter (inf on a miss).

        Newton-refines from _t_guess; surfaces whose guess is exact
        (_t_guess_is_exact) skip the iteration and return it directly.
        """
        t_init = self._t_guess(ray_origin, ray_direction)
        if self._t_guess_is_exact:
            return t_init
        t, _, _ = newton_raphson_intersect(self._sag_local, ray_origin, ray_direction, t_init)
        return t

    def compute_sag_and_normal_at(self, x, y):
        """(point, normal) at (x, y) for a single element, both (3,).

        The normal is unit length, obtained by autodiff of the sag.
        """
        z = self._sag_local(x, y)
        dzdx = jax.grad(self._sag_local, argnums=0)(x, y)
        dzdy = jax.grad(self._sag_local, argnums=1)(x, y)
        n = jnp.array([-dzdx, -dzdy, 1.0])
        point = jnp.stack([x, y, z], axis=-1)
        return point, n / jnp.linalg.norm(n)

    def sag_at(self, element_idx, x, y):
        """Sag z(x, y) of one element, relative to its decenter.

        Local frame. Used by the visualization module for mesh generation.
        """
        return self._index(element_idx)._sag_local(x, y)

    def intersect_t_at(self, element_idx, ray_origin, ray_direction):
        """Ray parameter and local landing point, without evaluating the surface.

        Returns (t, x, y): the intersection distance (inf on a miss)
        and the local in-surface coordinates of the landing point. On a miss
        the coordinates are those of the ray's closest approach, so they stay
        finite for downstream masking.
        """
        elem = self._index(element_idx)
        t = elem._intersect_t(ray_origin, ray_direction)
        t_safe = jnp.where(jnp.isfinite(t), t, 0.0)
        origin, t_offset, _ = _localize(ray_origin, ray_direction, jnp.zeros(3))
        hit = origin + (t_safe - t_offset) * ray_direction
        return t, hit[0], hit[1]

    def intersect_at(self, element_idx, ray_origin, ray_direction):
        """Intersect a local-frame ray with one element: (t, point, normal).

        Generic over the surface type -- _intersect_t resolves the ray
        parameter (closed-form for pure conics, Newton-refined otherwise) and
        point / normal follow from the sag at the hit. On a miss
        (t = inf) both are evaluated at the ray origin, so they stay finite
        for downstream masking.
        """
        t, x, y = self.intersect_t_at(element_idx, ray_origin, ray_direction)
        point, normal = self._index(element_idx).compute_sag_and_normal_at(x, y)
        return t, point, normal


# Composite (sum) surface


class SumSurfaceGroup(SurfaceGroup):
    """Composite surface whose sag is the sum of its components' sags.

    The composite's own offsets decenter the whole patch; component offsets
    (usually zero) are applied first, inside each component's _sag_local.

    Attributes
    ----------
    components
        Tuple of component SurfaceGroup instances, each
        sized to the same N.
    offsets : array, shape (N, 2)
        Per-element in-surface decenter for the composite.
        (inherited).
    """

    components: tuple

    def __init__(self, components, offsets=None):
        comps = tuple(components)
        if not comps:
            raise ValueError("SumSurfaceGroup requires at least one component")
        n = comps[0].offsets.shape[0]
        for c in comps:
            if c.offsets.shape[0] != n:
                raise ValueError(
                    "all components must describe the same number of elements N; "
                    f"got {[int(c.offsets.shape[0]) for c in comps]}"
                )
        self.components = comps
        self.offsets = _default_offsets(n, offsets)

    def _sag_intrinsic(self, x, y):
        total = self.components[0]._sag_local(x, y)
        for c in self.components[1:]:
            total = total + c._sag_local(x, y)
        return total

    def _t_guess(self, ray_origin, ray_direction):
        # Delegate to the base term (first component) for the analytic guess.
        return self.components[0]._t_guess(ray_origin, ray_direction)


# Aspheric helpers


def sag_raw(x, y, curvature, conic, aspheric):
    """Sag z(x, y) of the standard even asphere, without any decenter:

        z(r) = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2))
               + A4 r^4 + A6 r^6 + A8 r^8 + ...

    curvature is c (1/radius), conic is k, and aspheric is
    [A4, A6, ...] of shape (K,).
    """
    r2 = x * x + y * y
    c = curvature
    k = conic

    denom = 1 + jnp.sqrt(1 - (1 + k) * c * c * r2)
    z = r2 * c / denom

    if aspheric.size > 0:
        powers = jnp.arange(2, 2 + len(aspheric))
        z = z + jnp.sum(aspheric * r2**powers)

    return z


# --- Aspheric surface group ---


class AsphericSurfaceGroup(SurfaceGroup):
    """Batched aspheric surface parameters for N optical elements.

    A conic + even-polynomial surface used by OpticalElementGroup.
    When sliced/vmapped to a single element, each becomes a scalar-parameter
    surface; the generic SurfaceGroup machinery then handles sag,
    normal, and intersection. The conic provides a closed-form intersection
    initial guess via _t_guess.

    Attributes
    ----------
    curvatures : array, shape (N,)
        Per-element curvatures.
    conics : array, shape (N,)
        Per-element conic constants.
    aspherics : array, shape (N, K)
        Per-element even aspheric coefficients [A4, A6, ...]; column i
        multiplies r^(2i + 4). See sag_raw.
    offsets : array, shape (N, 2)
        Per-element in-surface decenter (inherited).
    """

    curvatures: jax.Array  # (N,)
    conics: jax.Array      # (N,)
    aspherics: jax.Array   # (N, K)

    def _sag_intrinsic(self, x, y):
        return sag_raw(x, y, self.curvatures, self.conics, self.aspherics)

    @property
    def _t_guess_is_exact(self) -> bool:
        return self.aspherics.shape[-1] == 0

    def _t_guess(self, ray_origin, ray_direction):
        c = self.curvatures
        k = self.conics
        a = self.aspherics
        x0 = self.offsets[0]
        y0 = self.offsets[1]

        # Translate the ray into the raw (unshifted) surface frame so the conic
        # closed-form guess is taken on the parent surface.
        z0 = sag_raw(x0, y0, c, k, a)
        ray_origin_raw = ray_origin + jnp.array([x0, y0, z0])
        return intersect_conic(ray_origin_raw, ray_direction, c, k)


# Non-batched sag/normal evaluation


def _single_asphere(offset, curvature, conic, aspheric) -> AsphericSurfaceGroup:
    """Single-element AsphericSurfaceGroup, sliced back down to scalar parameters."""
    return AsphericSurfaceGroup(
        offsets=jnp.asarray(offset).reshape(1, 2),
        curvatures=jnp.asarray(curvature).reshape(1),
        conics=jnp.asarray(conic).reshape(1),
        aspherics=jnp.asarray(aspheric).reshape(1, -1),
    )._index(0)


def sag(x, y, offset, curvature, conic, aspheric):
    """Sag z(x, y) in local mirror coordinates, relative to offset."""
    return _single_asphere(offset, curvature, conic, aspheric)._sag_local(x, y)


def compute_sag_and_normal(x, y, offset, curvature, conic, aspheric):
    """(point, normal) at (x, y), both (3,), from loose parameters."""
    return _single_asphere(offset, curvature, conic, aspheric).compute_sag_and_normal_at(x, y)


# --- Zernike ---

N_ZERNIKE = 11


def zernike_terms(u, v):
    """RMS-normalized Noll Zernike polynomials Z1..Z11.

    Evaluated in normalized Cartesian coordinates u = x / r_norm and
    v = y / r_norm (so the unit disk is u^2 + v^2 <= 1). The terms are
    written as Cartesian polynomials rather than via (rho, phi) so the
    gradient is smooth everywhere, including the origin.

    With the Noll RMS normalization each term has unit RMS over the unit disk,
    so a coefficient in metres equals that aberration's RMS surface contribution
    in metres. If your surface is not circular, you have to rescale.

    Parameters
    ----------
    u : scalar or array
        Normalized x-coordinate.
    v : scalar or array
        Normalized y-coordinate.

    Returns
    -------
    Array with the 11 lowest Noll terms stacked on the last axis, in Noll
    order
        piston, tilt x/y, defocus, astigmatism (oblique/vertical), coma
    (vertical/horizontal), trefoil (vertical/oblique), primary spherical.
    """
    r2 = u * u + v * v
    s3 = jnp.sqrt(3.0)
    s5 = jnp.sqrt(5.0)
    s6 = jnp.sqrt(6.0)
    s8 = jnp.sqrt(8.0)
    return jnp.stack(
        [
            jnp.ones_like(u),  # Z1  piston
            2.0 * u,  # Z2  tilt (x)
            2.0 * v,  # Z3  tilt (y)
            s3 * (2.0 * r2 - 1.0),  # Z4  defocus
            s6 * (2.0 * u * v),  # Z5  oblique astigmatism
            s6 * (u * u - v * v),  # Z6  vertical astigmatism
            s8 * (3.0 * r2 - 2.0) * v,  # Z7  vertical coma
            s8 * (3.0 * r2 - 2.0) * u,  # Z8  horizontal coma
            s8 * (3.0 * u * u * v - v * v * v),  # Z9  vertical trefoil
            s8 * (u * u * u - 3.0 * u * v * v),  # Z10 oblique trefoil
            s5 * (6.0 * r2 * r2 - 6.0 * r2 + 1.0),  # Z11 primary spherical
        ],
        axis=-1,
    )


class ZernikeSurfaceGroup(SurfaceGroup):
    """Standalone Zernike figure surface for N optical elements.

    Represents a surface whose height is a sum of RMS-normalized Noll Zernike
    polynomials, independent of any conic/aspheric base. Use it on its own to
    describe a pure figure-error surface, or as a term inside a
    SumSurfaceGroup to add a measured / random figure error on top of
    another surface.

    The normal is obtained by autodiff of the sag (inherited from
    SurfaceGroup), and the intersection uses the inherited tangent-plane
    initial guess.

    Attributes
    ----------
    coeffs : array, shape (N, J)
        Per-element Noll coefficients in metres. With J <= 11. Column m is Noll index m
        + 1 (Z1 = piston).
    r_norm : array, shape (N,)
        Per-element normalization radius in metres.
        rho = 1 at this radius.
    offsets : array, shape (N, 2)
        Per-element in-surface decenter (inherited).
    """

    coeffs: jax.Array  # (N, J)
    r_norm: jax.Array  # (N,)

    def __init__(self, coeffs, r_norm, offsets=None):
        coeffs = jnp.asarray(coeffs)
        if coeffs.ndim != 2:
            raise ValueError(f"coeffs must be 2D (N, J), got shape {coeffs.shape}")
        j = coeffs.shape[-1]
        if j > N_ZERNIKE:
            raise ValueError(
                f"coeffs provides {j} Noll terms, but only {N_ZERNIKE} are implemented (Z1..Z11)"
            )
        n = coeffs.shape[0]
        self.coeffs = coeffs
        self.r_norm = jnp.asarray(r_norm)
        self.offsets = _default_offsets(n, offsets)

    def _sag_intrinsic(self, x, y):
        u = x / self.r_norm
        v = y / self.r_norm
        terms = zernike_terms(u, v)  # (..., 11)
        j = self.coeffs.shape[-1]
        return jnp.sum(self.coeffs * terms[..., :j], axis=-1)


# Freeform surface


def _catmull_rom(p0, p1, p2, p3, t):
    """1D Catmull-Rom cardinal cubic at parameter t in [0, 1].

    Interpolates the middle interval [p1, p2] using the four samples
    p0..p3.
    """
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def bicubic_interp(grid, u, v):
    """Bicubic (Catmull-Rom) interpolation of an (H, W) height grid.

    grid[j, i] is the height at column i (x) and row j (y); u
    and v are fractional column / row coordinates. Exact at grid nodes.
    Queries outside the grid clamp to the edge (flat extrapolation), which
    keeps the Newton intersection well-behaved when it strays off the patch.
    """
    h, w = grid.shape
    u = jnp.clip(u, 0.0, w - 1.0)
    v = jnp.clip(v, 0.0, h - 1.0)

    # Integer cell, kept one short of the far edge so the [i, i+1] span is valid.
    # floor / int cast carry no gradient, so the slope flows through fu / fv.
    uf = jnp.clip(jnp.floor(u), 0.0, w - 2.0)
    vf = jnp.clip(jnp.floor(v), 0.0, h - 2.0)
    i = uf.astype(jnp.int32)
    j = vf.astype(jnp.int32)
    fu = u - uf
    fv = v - vf

    cols = jnp.clip(i + jnp.arange(-1, 3), 0, w - 1)  # (4,) x neighbourhood
    rows = jnp.clip(j + jnp.arange(-1, 3), 0, h - 1)  # (4,) y neighbourhood
    patch = grid[rows][:, cols]  # (4, 4): rows over y, cols over x

    # Interpolate along x for each of the four rows, then along y.
    row_vals = _catmull_rom(
        patch[:, 0],
        patch[:, 1],
        patch[:, 2],
        patch[:, 3],
        fu,
    )
    return _catmull_rom(
        row_vals[0],
        row_vals[1],
        row_vals[2],
        row_vals[3],
        fv,
    )


class FreeformSurfaceGroup(SurfaceGroup):
    """Per-element freeform surface defined by a bicubically interpolated grid.

    Each element carries a regular (H, W) height map sampled over a
    rectangular domain; the sag at arbitrary (x, y) is the Catmull-Rom
    bicubic interpolation of that map.

    For a strongly curved freeform, compose it on top of an
    AsphericSurfaceGroup in a SumSurfaceGroup (base term first)
    so the conic supplies the intersection initial guess.

    Attributes
    ----------
    grid_z : array, shape (N, H, W)
        Per-element height samples in metres. grid_z[n, j, i] is the height
        of element n at column i (x), row j (y).
    x0, y0 : array, shape (N,)
        Per-element grid origin — the coordinate of column /
        row 0.
    dx, dy : array, shape (N,)
        Per-element grid spacing along x / y.
    offsets : array, shape (N, 2)
        Per-element in-surface decenter (inherited).
    """

    grid_z: jax.Array  # (N, H, W)
    x0: jax.Array      # (N,)
    y0: jax.Array      # (N,)
    dx: jax.Array      # (N,)
    dy: jax.Array      # (N,)

    def __init__(self, grid_z, x0, y0, dx, dy, offsets=None):
        grid_z = jnp.asarray(grid_z)
        if grid_z.ndim != 3:
            raise ValueError(f"grid_z must be 3D (N, H, W), got shape {grid_z.shape}")
        n, h, w = grid_z.shape
        if h < 2 or w < 2:
            raise ValueError(f"grid must be at least 2x2 per element, got ({h}, {w})")
        self.grid_z = grid_z
        self.x0 = jnp.broadcast_to(jnp.asarray(x0, dtype=grid_z.dtype), (n,))
        self.y0 = jnp.broadcast_to(jnp.asarray(y0, dtype=grid_z.dtype), (n,))
        self.dx = jnp.broadcast_to(jnp.asarray(dx, dtype=grid_z.dtype), (n,))
        self.dy = jnp.broadcast_to(jnp.asarray(dy, dtype=grid_z.dtype), (n,))
        self.offsets = _default_offsets(n, offsets)

    @classmethod
    def from_extent(cls, grid_z, half_width, half_height, offsets=None):
        """Build from a grid centred on the origin spanning a given extent.

        The grid columns span [-half_width, half_width] and rows span
        [-half_height, half_height]. half_width / half_height may be
        scalar (shared) or per-element (N,).
        """
        grid_z = jnp.asarray(grid_z)
        if grid_z.ndim != 3:
            raise ValueError(f"grid_z must be 3D (N, H, W), got shape {grid_z.shape}")
        n, h, w = grid_z.shape
        hw = jnp.broadcast_to(jnp.asarray(half_width, dtype=grid_z.dtype), (n,))
        hh = jnp.broadcast_to(jnp.asarray(half_height, dtype=grid_z.dtype), (n,))
        return cls(
            grid_z=grid_z,
            x0=-hw,
            y0=-hh,
            dx=2.0 * hw / (w - 1),
            dy=2.0 * hh / (h - 1),
            offsets=offsets,
        )

    def _sag_intrinsic(self, x, y):
        u = (x - self.x0) / self.dx
        v = (y - self.y0) / self.dy
        return bicubic_interp(self.grid_z, u, v)
