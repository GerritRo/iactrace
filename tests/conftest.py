import jax
import pytest


@pytest.fixture(autouse=True)
def set_jax_precision():
    """Set to float32 by standard, since this is jax default"""
    jax.config.update("jax_enable_x64", False)
    yield


@pytest.fixture
def random_key():
    """Provide a fresh random key for tests that need randomness."""
    return jax.random.key(42)


@pytest.fixture
def n_samples():
    """Small primary-sample count for fast I/O round-trip tests."""
    return 4
