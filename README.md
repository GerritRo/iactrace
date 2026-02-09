# IACTrace

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](licenses/LICENSE.rst)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/GerritRo/iactrace/branch/main/graph/badge.svg)](https://codecov.io/gh/GerritRo/iactrace)

**JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes**

IACTrace is a differentiable ray tracing library for simulating the optical properties of IACT (Imaging Atmospheric Cherenkov Telescope) systems. Built on JAX and Equinox, it supports automatic differentiation for optimization and inverse problems.

## Features

- Differentiable ray tracing with JAX
- Multi-stage optical systems (primary, secondary mirrors)
- Square and hexagonal sensor arrays
- Aspheric mirror surfaces with configurable parameters
- Obstruction modeling (cylinders, boxes, spheres)
- YAML-based telescope configuration
- Response matrix calculation

## Installation

```bash
pip install git+https://github.com/GerritRo/iactrace/
```

For development:
```bash
git clone https://github.com/GerritRo/iactrace.git
cd iactrace
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from iactrace import MCIntegrator, load_telescope

# Load telescope from YAML configuration
key = jax.random.key(0)
integrator = MCIntegrator(n_samples=128)
telescope = load_telescope("configs/HESS/CT3.yaml", integrator, key)

# Define point sources
n_sources = 3
key, key1, key2 = jax.random.split(key, 3)

x = jax.random.uniform(key1, (n_sources,), minval=-1, maxval=1)
y = jax.random.uniform(key2, (n_sources,), minval=-1, maxval=1)
z = jnp.ones(n_sources) * 500  # Distance in meters

sources = jnp.stack([x, y, z], axis=1)  # (N, 3) positions
values = jnp.ones(n_sources)             # (N,) intensities

# Render image
image = telescope.render(sources, values, source_type='point')
```

## Documentation

Full documentation and examples are available at: **https://gerritro.github.io/iactrace/**

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

BSD-3-Clause License - see [LICENSE](licenses/LICENSE.rst) for details.

## Citation

If you use IACTrace in your research, please cite this repository.