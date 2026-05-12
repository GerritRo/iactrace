Installation
============

Requirements
------------

IACTrace requires **Python 3.12 or later**.

For GPU acceleration, you will also need:

- NVIDIA GPU with CUDA support
- CUDA toolkit (version 11.8 or 12.x recommended)
- cuDNN

Basic Installation
------------------

Install directly from the git repository:

.. code-block:: bash

   pip install git+https://github.com/GerritRo/iactrace.git

Development Installation
------------------------

For development or to access example notebooks, clone the repository:

.. code-block:: bash

   git clone https://github.com/GerritRo/iactrace.git
   cd iactrace
   pip install -e ".[dev]"

This installs additional development dependencies:

- ``pytest`` for running tests
- ``jupyter`` for example notebooks
- ``mypy`` for type checking
- ``ruff`` for code linting

GPU Support
-----------

By default, JAX installs with CPU-only support. For GPU acceleration, install
the appropriate JAX version for your CUDA installation:

.. code-block:: bash

   # For CUDA 12
   pip install --upgrade "jax[cuda12]"

   # For CUDA 11
   pip install --upgrade "jax[cuda11_local]"

See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
for detailed instructions and troubleshooting.

Verify your installation:

.. code-block:: python

   import jax
   print(jax.devices())  # Should show GPU device if available

Building Documentation
----------------------

To build this documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Dependencies
------------

IACTrace depends on:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Package
     - Version
     - Purpose
   * - `JAX <https://jax.readthedocs.io/>`_
     - >= 0.4.20
     - Numerical computing and automatic differentiation
   * - `Equinox <https://docs.kidger.site/equinox/>`_
     - >= 0.11.0
     - PyTree-based neural network library for JAX
   * - `NumPy <https://numpy.org/>`_
     - >= 1.22.0
     - Array operations and interoperability
   * - `Matplotlib <https://matplotlib.org/>`_
     - >= 3.5.0
     - 2D plotting and visualization
   * - `Trimesh <https://trimsh.org/>`_
     - >= 3.15.0
     - 3D geometry and visualization
   * - `PyYAML <https://pyyaml.org/>`_
     - >= 6.0
     - YAML configuration parsing

Troubleshooting
---------------

**JAX not finding GPU**

Ensure CUDA is properly installed and visible:

.. code-block:: bash

   nvidia-smi  # Should show GPU info

If JAX still uses CPU, check that you installed the correct JAX version for
your CUDA version.

**Memory errors with large telescopes**

For telescopes with many facets or high ``n_samples``, you may encounter memory
issues. Solutions:

- Reduce ``n_samples`` for initial testing
- Process in chunks with different random keys

**Slow first execution**

JAX compiles functions on first call. This is normal and subsequent calls will
be fast. To avoid recompilation, structure your code to reuse compiled
functions.