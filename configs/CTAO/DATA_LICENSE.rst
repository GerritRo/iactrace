CTAO-like Telescope Configuration Data
======================================

License
-------

All files in this folder are licensed under
`CC-BY-4.0 <https://creativecommons.org/licenses/by/4.0/>`_.

Source
------

The files are derived from the CTAO Simulation Telescope Models for CORSIKA
and sim_telarray, Prod6 v1.0, published by the CTA Observatory under
CC-BY-4.0.

- **Original source**: https://doi.org/10.5281/zenodo.14198379
- **Citation**: CTA Observatory. CTAO Simulation Telescope Models for CORSIKA 
  and sim_telarray - Prod6. 1.0, Zenodo, 21 Nov. 2024,
  https://doi.org/10.5281/zenodo.14198379.

The camera pixel layouts were read from the camera geometry tables of ctapipe.

- **Citation**: Maximilian Linhoff et al. Cta-observatory/ctapipe: V0.30.0 –
  2026-03-20. v0.30.0, Zenodo, 20 Mar. 2026,
  https://doi.org/10.5281/zenodo.19134070.

Obstruction and support geometry are not accurate, but reproduced from publically 
available images.

Changes
-------

These files are not copies of the Prod6 configuration. They were produced in
three ways:

- **Extracted**: values read from the Prod6 files and converted to YAML, with
  units and coordinate conventions changed to those of iactrace.
- **Simplified**: structures that Prod6 does not describe as geometry, or
  describes in more detail than is used here, replaced by simple shapes.
- **Fitted or modelled**: parameters with no direct counterpart in Prod6,
  chosen so that the model reproduces a Prod6 table.

These are approximate, "CTAO-like" models made for iactrace. They are not
official CTAO telescope models.

Acknowledgement
---------------

The CTAO asks that work using the Prod6 telescope models carries the following
acknowledgement. It applies to work using the files in this folder as well:

    This research has made use of the CTAO Simulation Telescope Models
    provided by the CTAO Central Organisation and Consortium (version prod6
    v1.0; https://doi.org/10.5281/zenodo.14198379).

Disclaimer
----------

This material is provided "as-is" without warranties. See CC-BY-4.0
`Section 5 <https://creativecommons.org/licenses/by/4.0/legalcode#s5>`_
for full disclaimer of warranties and limitation of liability.
