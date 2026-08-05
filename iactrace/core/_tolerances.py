from __future__ import annotations

import jax.numpy as jnp

_DIR_FACTOR = 8.0
_LEN_FACTOR = 32.0

def eps(x) -> float:
    return float(jnp.finfo(jnp.result_type(x, jnp.float32)).eps)

def dir_tol(x) -> float:
    return _DIR_FACTOR * eps(x)

def len_rel(x) -> float:
    return _LEN_FACTOR * eps(x)