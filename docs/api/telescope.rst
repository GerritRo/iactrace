iactrace.telescope
==================

The Telescope class and helper modules for building optical systems.

Telescope Class
---------------

.. autoclass:: iactrace.telescope.Telescope
   :members:
   :undoc-members:

Optical Element Composition
---------------------------

Mirrors and lenses are both represented as
:class:`~iactrace.core.OpticalElementGroup` instances composed from a
surface, an aperture, an interaction, and an optional BSDF (see
:doc:`core`). The helper submodules below provide builders that assemble
the right combination for common optical elements.

Mirror builders
~~~~~~~~~~~~~~~

.. automodule:: iactrace.telescope.mirrors
   :members:

Lens builders
~~~~~~~~~~~~~

.. automodule:: iactrace.telescope.lenses
   :members:

Obstruction builders
~~~~~~~~~~~~~~~~~~~~

.. automodule:: iactrace.telescope.obstructions
   :members:

Operations
----------

Functional operations for modifying telescope configurations (e.g., for
calibration, error injection, optimization). All operations return new
telescope instances.

.. automodule:: iactrace.telescope.operations
   :members:
   :undoc-members:
