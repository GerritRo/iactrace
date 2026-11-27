from __future__ import annotations

import jax.numpy as jnp


def bracket(grid, x):
    """Grid interval holding each x: (lower index, upper index, weight of the upper).

    The building block of a table lookup, clamped at the grid edges like
    jax.numpy.interp; a single-node grid gives that node with weight 0. The
    search compares against every node instead of bisecting, which keeps a
    loop out of the compiled trace -- on a GPU that loop costs far more than
    the lookup itself.
    """
    last = grid.shape[0] - 1
    index = jnp.searchsorted(grid, x, side="right", method="compare_all") - 1
    lo = jnp.clip(index, 0, max(last - 1, 0))
    hi = jnp.minimum(lo + 1, last)
    span = grid[hi] - grid[lo]
    weight = (x - grid[lo]) / jnp.where(span > 0, span, 1.0)
    return lo, hi, jnp.clip(weight, 0.0, 1.0)
