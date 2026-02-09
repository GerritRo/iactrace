iactrace.telescope
==================

The Telescope class and related components for building optical systems.

Telescope Class
---------------

The main class representing an optical telescope system:

.. autoclass:: iactrace.telescope.Telescope
   :members:
   :undoc-members:
   :show-inheritance:

Mirror Groups
-------------

Classes for representing groups of mirror facets:

.. autoclass:: iactrace.telescope.MirrorGroup
   :members:
   :undoc-members:

.. autoclass:: iactrace.telescope.AsphericDiskMirrorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.telescope.AsphericPolygonMirrorGroup
   :members:
   :undoc-members:
   :show-inheritance:

Lens Groups
-----------

Classes for representing refractive optical elements:

.. autoclass:: iactrace.telescope.LensGroup
   :members:
   :undoc-members:

.. autoclass:: iactrace.telescope.AsphericDiskLensGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iactrace.telescope.PlanoSlabGroup
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Functional operations for modifying telescope configurations (e.g., for optimization):

.. automodule:: iactrace.telescope.operations
   :members:
   :undoc-members: