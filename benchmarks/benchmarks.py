from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from iactrace import Telescope

# Get absolute path to config files (relative to this benchmark file)
_BENCHMARK_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _BENCHMARK_DIR.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "HESS" / "CT3.yaml"


def _block(result):
    """Block until JAX computation is complete and return result.

    This ensures accurate timing by waiting for async JAX operations.
    """
    if isinstance(result, tuple):
        return tuple(jax.block_until_ready(r) for r in result)
    return jax.block_until_ready(result)


class RenderBenchmarks:
    """Benchmarks for the render() function.

    This is the main user-facing function for tracing sources through
    a telescope onto a sensor.
    """

    params = [
        [1, 10],  # n_sources
        [16, 256],  # n_samples (integrator samples per element)
    ]
    param_names = ["n_sources", "n_samples"]

    timeout = 300

    def setup(self, n_sources, n_samples):
        """Set up telescope and sources, trigger JIT compilation."""
        self.telescope = Telescope.from_yaml(
            str(_CONFIG_PATH),
            n_samples=n_samples,
            key=jax.random.key(42),
        )

        # Create point sources at varying positions
        key = jax.random.key(123)
        self.sources = jax.random.uniform(
            key,
            (n_sources, 3),
            minval=jnp.array([-5.0, -5.0, 1000.0]),
            maxval=jnp.array([5.0, 5.0, 2000.0]),
        )
        self.values = jnp.ones(n_sources)

        # Warmup: trigger JIT compilation. Telescope.render returns a
        # LazyRayBundle, so we have to materialise it to actually execute
        # the ray-trace.
        _ = _block(
            self.telescope.render(self.sources, self.values, source_type="point").materialise()
        )

    def time_render_point_sources(self, n_sources, n_samples):
        """Time rendering point sources."""
        result = self.telescope.render(self.sources, self.values, source_type="point").materialise()
        _block(result)


class RayTracingBenchmarks:
    """Benchmarks for trace_rays() - classical ray tracing.

    Unlike render() which samples from primary mirror surfaces,
    trace_rays() traces rays from arbitrary origins.
    """

    params = [[100, 1000]]
    param_names = ["n_rays"]

    timeout = 300

    def setup(self, n_rays):
        """Set up telescope and rays, trigger JIT compilation."""
        self.telescope = Telescope.from_yaml(
            str(_CONFIG_PATH),
            n_samples=1,
            key=jax.random.key(42),
        )

        # Create rays starting above the telescope, pointing down
        key = jax.random.key(789)
        key1, key2 = jax.random.split(key)

        # Random positions in a disk above the telescope
        r = 5.0 * jnp.sqrt(jax.random.uniform(key1, (n_rays,)))
        theta = jax.random.uniform(key2, (n_rays,)) * 2 * jnp.pi

        self.origins = jnp.stack(
            [r * jnp.cos(theta), r * jnp.sin(theta), jnp.ones(n_rays) * 100.0], axis=1
        )

        self.directions = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n_rays, 3))
        self.values = jnp.ones(n_rays)

        # Warmup: trigger JIT compilation
        _ = _block(self.telescope.trace(self.origins, self.directions, self.values))

    def time_trace_rays(self, n_rays):
        """Time classical ray tracing."""
        result = self.telescope.trace(self.origins, self.directions, self.values)
        _block(result)


class GradientBenchmarks:
    """Benchmarks for gradient computation through the ray tracer.

    iactrace is differentiable, so we measure the cost of computing
    gradients with respect to telescope parameters.
    """

    params = [
        [1, 10],  # n_sources
        [16, 64],  # n_samples
    ]
    param_names = ["n_sources", "n_samples"]

    timeout = 600

    def setup(self, n_sources, n_samples):
        """Set up telescope and sources, trigger JIT compilation."""
        telescope = Telescope.from_yaml(
            str(_CONFIG_PATH),
            n_samples=n_samples,
            key=jax.random.key(42),
        )

        key = jax.random.key(111)
        self.sources = jax.random.uniform(
            key,
            (n_sources, 3),
            minval=jnp.array([-2.0, -2.0, 1000.0]),
            maxval=jnp.array([2.0, 2.0, 1500.0]),
        )
        self.values = jnp.ones(n_sources)

        # Split telescope into trainable (arrays) and static (non-arrays) parts
        filter_spec = jax.tree.map(eqx.is_array, telescope)
        self.trainable, self.static = eqx.partition(telescope, filter_spec)

        # Define loss function that takes trainable/static split. We sum
        def loss_fn(trainable, static):
            tel = eqx.combine(trainable, static)
            rays = tel.render(
                self.sources,
                self.values,
                source_type="point",
            ).materialise()
            return jnp.sum(rays.directions**2)

        self.loss_fn = loss_fn
        self.grad_fn = eqx.filter_value_and_grad(loss_fn)

        # Warmup: trigger JIT compilation for both forward and backward
        _ = _block(self.loss_fn(self.trainable, self.static))
        _ = _block(self.grad_fn(self.trainable, self.static))

    def time_gradient(self, n_sources, n_samples):
        """Time gradient computation (includes forward pass)."""
        result = self.grad_fn(self.trainable, self.static)
        _block(result)


class TelescopeLoadingBenchmarks:
    """Benchmarks for telescope configuration loading.

    These measure the one-time setup cost of loading telescopes.
    """

    params = [[256]]
    param_names = ["n_samples"]

    timeout = 120

    def time_load_from_yaml(self, n_samples):
        """Time loading telescope from YAML configuration."""
        tel = Telescope.from_yaml(
            str(_CONFIG_PATH),
            n_samples=n_samples,
            key=jax.random.key(42),
        )
        # Force evaluation of lazy operations
        _ = tel.mirror_groups
