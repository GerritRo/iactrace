iactrace.io
===========

Functions for loading and saving telescope and camera configurations.

Loaders
-------

.. autofunction:: iactrace.io.load_telescope_config

.. autofunction:: iactrace.io.load_camera_config

Builders
--------

.. autofunction:: iactrace.io.build_telescope_config

.. autofunction:: iactrace.io.build_camera_config

Savers
------

.. autofunction:: iactrace.io.save_telescope

.. autofunction:: iactrace.io.save_camera

.. autofunction:: iactrace.io.telescope_to_dict

.. autofunction:: iactrace.io.camera_to_dict

Schemas
-------

.. autoclass:: iactrace.io.TelescopeConfigSchema
   :members:

.. autoclass:: iactrace.io.CameraFileSchema
   :members:

Exceptions
----------

.. autoexception:: iactrace.io.YAMLConfigError
   :show-inheritance:
