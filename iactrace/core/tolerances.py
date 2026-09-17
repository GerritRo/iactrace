from __future__ import annotations

import jax.numpy as jnp

_DIR_FACTOR = 8.0
_LEN_FACTOR = 32.0


def _eps(x) -> float:
    return float(jnp.finfo(jnp.result_type(x, jnp.float32)).eps)


def dir_tol(x) -> float:
    return _DIR_FACTOR * _eps(x)


def len_rel(x) -> float:
    return _LEN_FACTOR * _eps(x)
