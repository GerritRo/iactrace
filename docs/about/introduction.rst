Introduction
============

IACTrace is a high-performance, differentiable ray tracing library for simulating
the optical properties of IACT (Imaging Atmospheric Cherenkov Telescope) systems.
Built on JAX and Equinox, it supports automatic differentiation for optimization and inverse problems.

What is IACTrace?
-----------------

IACTrace provides a computational framework for:

- **IACT optical simulation**: Trace light rays through complex telescope geometries
  including segmented primary mirrors, secondary mirrors, and obstruction geometry.

- **Differentiable simulation**: Compute gradients through the entire simulation
  pipeline, enabling gradient-based optimization of telescope parameters.
  
- **Fast Rendering**: Simplifications for parallel light and point sources enable
  fast simulation of skyfields, especially with GPU support.

- **Response characterization**: Generate response matrices that map sky
  positions to camera pixels, essential for `nyx <https://github.com/GerritRo/nyx>`_.

Target Audience
---------------

This library is designed for:

- **Calibration specialists** especially in the context of solving inverse problems
- **Python extremists** looking for a pip-installable python-native package
- **Speed freaks** dissatisfied with the speed of the alternatives

The library assumes familiarity with Python, NumPy-style array programming, and
basic optics concepts. Experience with JAX is helpful but not required.

License
-------

IACTrace is released under the BSD-3-Clause license. See the
`LICENSE <https://github.com/GerritRo/iactrace/-/blob/main/licenses/LICENSE.rst>`_
file for details.

Citation
--------

If you use IACTrace in your research, please cite::

   @software{iactrace,
     author = {Roellinghoff, Gerrit},
     title = {IACTrace: Differentiable Ray Tracing for IACTs},
     url = {https://github.com/GerritRo/iactrace},
     year = {2025}
   }