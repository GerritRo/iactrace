Wavelength and Response Curves
==============================

Every ray carries a wavelength (``RayBundle.wavelength``, in nanometres by
convention, ``400.0`` when unspecified). Three things read it:

============================  ==========================================
What reads it                 How you set it
============================  ==========================================
The source                    a :class:`~iactrace.core.Spectrum` on ``render``
Reflectivity, transmittance   a :class:`~iactrace.core.ResponseCurve` on the element
Quantum efficiency            a :class:`~iactrace.core.ResponseCurve` on the detector
Refraction                    a :class:`~iactrace.core.RefractiveIndex` on the element
============================  ==========================================

Leave all three unset and every ray runs at 400 nm with flat coefficients.

Source spectra
--------------

``wavelength`` on :meth:`~iactrace.Telescope.render` takes a scalar or a
:class:`~iactrace.core.Spectrum`; :meth:`~iactrace.Telescope.trace` also accepts a
per-ray ``(N,)`` array, since you supply its rays.

.. code-block:: python

   import jax.numpy as jnp
   from iactrace import TabulatedSpectrum

   # Monochromatic.
   image = camera.image(telescope.render(dirs, vals, "parallel", wavelength=450.0))

   # Cherenkov-like: photon density proportional to 1 / lambda^2.
   wl = jnp.linspace(300.0, 600.0, 31)
   cherenkov = TabulatedSpectrum.from_density(wl, 1.0 / wl**2)

   image = camera.image(telescope.render(dirs, vals, "parallel", wavelength=cherenkov))

The draw is reparameterised, so gradients flow to the spectrum's own
parameters. For a deterministic answer instead, sweep the quadrature nodes
from :meth:`~iactrace.core.Spectrum.bins`:

.. code-block:: python

   wavelengths, weights = cherenkov.bins()
   image = sum(
       float(w) * camera.image(telescope.render(dirs, vals, "parallel", wavelength=float(l)))
       for l, w in zip(wavelengths, weights)
   )

Response curves
---------------

A :class:`~iactrace.core.ResponseCurve` maps a ray's incidence angle and
wavelength to a coefficient in [0, 1]. Every element that attenuates light
takes the same **bulk scalar × curve** pair:

=====================================  ====================  ==========================
Element                                Bulk                  Curve
=====================================  ====================  ==========================
Mirror facet                           ``reflectivity``      ``reflectivity_curve``
Lens / window                          ``transmittance``     ``transmittance_curve``
Winston / Okumura cone (per bounce)    ``reflectivity``      ``reflectivity_curve``
Photodetector                          ``qe``                ``qe_curve``
=====================================  ====================  ==========================

Build one with :meth:`~iactrace.core.TabulatedResponse.from_degrees` for an angle
(or angle x wavelength) table, or
:meth:`~iactrace.core.TabulatedResponse.from_wavelengths` for the angle-flat
:math:`R(\lambda)` case:

.. code-block:: python

   from iactrace import TabulatedResponse

   # R(lambda), same at every angle.
   refl = TabulatedResponse.from_wavelengths(
       [300.0, 400.0, 550.0], [0.70, 0.90, 0.85], n_elements=1,
   )

   # R(theta, lambda): one row per angle, one column per wavelength.
   coating = TabulatedResponse.from_degrees(
       angles_deg=[0.0, 45.0, 70.0],
       wavelengths=[300.0, 550.0],
       values=[[0.90, 0.92], [0.88, 0.90], [0.70, 0.75]],
       n_elements=1,
   )

Values are interpolated bilinearly and **clamped** at the grid edges, so a
table never extrapolates. ``n_elements`` is 1 for a curve shared by every
facet, or ``N`` to give each facet its own row.

For the common measured-QE case there is a shortcut that builds the detector
and its curve in one step:

.. code-block:: python

   from iactrace import TabulatedQE

   detector = TabulatedQE.from_table([280.0, 350.0, 450.0, 600.0],
                                     [0.15, 0.38, 0.32, 0.10])

.. note::

   On a refracting element the curve *replaces* the default Fresnel
   transmission rather than multiplying it. Leave
   ``transmittance_curve`` unset to keep Fresnel, which is computed from the
   element's index and therefore already dispersive.

Dispersion
----------

Any place an ``index`` is expected accepts either a plain number
(non-dispersive) or an :class:`~iactrace.core.RefractiveIndex` model:

.. code-block:: python

   from iactrace import SellmeierIndex, TabulatedIndex

   # Measured n(lambda) samples.
   glass = TabulatedIndex.from_table([300.0, 700.0], [1.55, 1.51], n_elements=1)

   # N-BK7, Sellmeier coefficients in nanometres.
   bk7 = SellmeierIndex(
       b=jnp.array([[1.03961212, 0.231792344, 1.01046945]]),
       c=jnp.array([[6000.69867, 20017.9144, 103560653.0]]),
   )

The Sellmeier ``c`` coefficients carry units of wavelength squared and must
match the ray wavelengths -- nm^2 for the nanometre convention used
throughout. The same field is used for a PMT's entrance window
(``window_index``), so a dispersive window gets a wavelength-dependent
Fresnel loss for free.

In YAML
-------

Both curve and index models round-trip through the config files. Curves are
``{type: table, ...}`` everywhere; indices are ``sellmeier`` or
``index_table``:

.. code-block:: yaml

   mirror_templates:
     primary:
       surface: {type: aspheric, curvature: 0.0333, conic: -1.0}
       reflectivity: 0.9              # bulk
       reflectivity_curve:            # R(theta, lambda), rows = angles
         type: table
         angles_deg: [0.0, 30.0, 45.0, 70.0]
         wavelengths_nm: [300.0, 400.0, 550.0]
         values:
           - [0.88, 0.92, 0.90]
           - [0.87, 0.91, 0.89]
           - [0.86, 0.90, 0.88]
           - [0.70, 0.75, 0.72]

   lenses:
     - type: plano_slab
       position: [0, 0, 14.0]
       orientation: [0, 0, 0]
       aperture: {type: circular, radius: 0.5}
       thickness: 0.005
       index:                          # or just: index: 1.52
         type: sellmeier
         b: [1.03961212, 0.231792344, 1.01046945]
         c: [6000.69867, 20017.9144, 103560653.0]
       stage: 1

Omitting ``wavelengths_nm`` makes the table angle-only; a single
``angles_deg: [0.0]`` row makes it wavelength-only. On the camera side the
same block appears as a concentrator's ``reflectivity_curve`` and a
detector's ``qe_curve``:

.. code-block:: yaml

   photodetector:
     type: pmt
     qe: 1.0
     qe_curve:                        # QE(lambda), angle-flat
       type: table
       angles_deg: [0.0]
       wavelengths_nm: [280.0, 350.0, 450.0, 600.0]
       values:
         - [0.15, 0.38, 0.32, 0.10]
     window_index: 1.52               # or a sellmeier / index_table block
     face_radius: 0.01925

Next Steps
----------

- :doc:`custom_telescopes` - the full YAML schema
- :doc:`/api/core` - ``Spectrum``, ``ResponseCurve`` and ``RefractiveIndex`` reference
