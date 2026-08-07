# IACTrace

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/GerritRo/iactrace/branch/main/graph/badge.svg)](https://codecov.io/gh/GerritRo/iactrace)

**JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes**

IACTrace is a differentiable ray tracing library for simulating the optical properties of IACT (Imaging Atmospheric Cherenkov Telescope) systems. Built on JAX and Equinox, it supports automatic differentiation for optimization and inverse problems.

## Features

- Differentiable ray tracing with JAX
- Multi-stage optical systems: segmented primaries, secondary mirrors, lenses and windows
- Square and hexagonal sensor arrays with per-pixel light concentrators (Winston / Okumura cones)
- Physical photodetectors: PMTs, SiPMs can be connected in-line with concentrators
- Aspheric, Zernike and freeform surfaces, plus error models (roughness, misalignment, figure errors)
- Obstruction modeling (cylinders, boxes, spheres, oriented boxes, triangles)
- YAML-based telescope and camera configuration
- Response matrix calculation

## Installation
```bash
pip install iactrace
```
Or straight from the repository:

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

Telescope and camera are configured with separate YAML files and loaded
independently.

```python
import jax
import jax.numpy as jnp
from iactrace import Telescope, Camera

# Random key for Monte Carlo ray sampling.
key = jax.random.key(0)

# Load the telescope (optics + camera frame) and the camera (sensor layout).
telescope = Telescope.from_yaml("configs/HESS/CT3.yaml", n_samples=256, key=key)
camera = Camera.from_yaml("configs/HESS/HESS1U.yaml")

# Simulate an on-axis astronomical source (parallel light).
directions = jnp.array([[0.0, 0.0, -1.0]])
values = jnp.array([1.0])

# render() traces rays through the optics and returns a lazy ray bundle;
# camera.image() folds it into the pixel image.
ray_bundle = telescope.render(directions, values, source_type="parallel")
image = camera.image(ray_bundle)
```

See the [Quick Start guide](https://gerritro.github.io/iactrace/getting_started/quickstart.html)
for point sources, off-axis geometry, and visualization.

## Documentation

Full documentation and examples are available at: **https://gerritro.github.io/iactrace/**

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

BSD-3-Clause License - see [LICENSE](LICENSE) for details.

## Citation

If you use IACTrace in your research, please cite this repository.
