iactrace.io
===========

Everything iactrace reads from or writes to a file: telescope and camera
configuration as YAML, and the result of a scan as ``.npz``.

For how the package is layered, and what to touch when adding a new schema
variant or a new file format, see ``iactrace/io/README.md``.

Configuration
-------------

Loaders
~~~~~~~

.. autofunction:: iactrace.io.load_telescope_config

.. autofunction:: iactrace.io.load_camera_config

Builders
~~~~~~~~

Take an already-parsed dictionary, for a caller that read the YAML itself.

.. autofunction:: iactrace.io.build_telescope_config

.. autofunction:: iactrace.io.build_camera_config

Savers
~~~~~~

.. autofunction:: iactrace.io.save_telescope

.. autofunction:: iactrace.io.save_camera

.. autofunction:: iactrace.io.telescope_to_dict

.. autofunction:: iactrace.io.camera_to_dict

Schemas
~~~~~~~

The grammar a config file is validated against.

.. autoclass:: iactrace.io.TelescopeConfigSchema
   :members:

.. autoclass:: iactrace.io.CameraFileSchema
   :members:

Exceptions
~~~~~~~~~~

.. autoexception:: iactrace.io.YAMLConfigError
   :show-inheritance:

Effective-aperture tables
-------------------------

.. automodule:: iactrace.io.aperture_table

.. autofunction:: iactrace.io.save_aperture_table

.. autofunction:: iactrace.io.load_aperture_table

.. autofunction:: iactrace.io.read_aperture_table_arrays

.. autodata:: iactrace.io.FORMAT

.. autodata:: iactrace.io.FORMAT_VERSION
