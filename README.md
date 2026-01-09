# IACTrace

**JAX-based optical ray tracing for Imaging Atmospheric Cherenkov Telescopes**

IACTrace is a high-performance, differentiable ray tracing library for simulating the optical properties of IACT (Imaging Atmospheric Cherenkov Telescope) systems. Built on JAX and Equinox, it supports automatic differentiation for optimization and inverse problems.

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
pip install iactrace
```

For GPU support:
```bash
pip install iactrace[gpu]
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
from iactrace import Telescope, MCIntegrator, load_telescope

# Load telescope from YAML configuration
key = jax.random.key(0)
integrator = MCIntegrator(n_samples=1000)
telescope = load_telescope("configs/HESS/CT3.yaml", integrator, key)

# Define point sources
sources = jax.numpy.array([[0.0, 0.0, 1e10]])  # (N, 3) positions
values = jax.numpy.array([1.0])                 # (N,) intensities

# Render image
image = telescope(sources, values, source_type='point')
```

## Documentation

See the `examples/` directory for Jupyter notebooks demonstrating:

- `HESS_I.ipynb` - H.E.S.S. telescope I simulation
- `HESS_II.ipynb` - H.E.S.S. telescope II simulation
- `Cassegrain.ipynb` - Cassegrain telescope example
- `ResponseMatrix.ipynb` - Response matrix calculation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use IACTrace in your research, please cite this repository.