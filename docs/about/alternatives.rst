Alternatives
============

IACTrace is not the only tool for simulating segmented telescope optics. This page
compares it to other commonly used approaches in the IACT community and beyond.

sim_telarray
----------------------

`sim_telarray <https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/>`_ is the
de facto standard for IACT simulations within the Cherenkov Telescope Array
Observatory (CTAO) and H.E.S.S. collaborations.

**Advantages of sim_telarray:**

- Mature, validated against real telescope data
- Integrated with CORSIKA for full air shower simulation
- Handles detector electronics and trigger logic
- Well-documented with established analysis pipelines

**When to use IACTrace instead:**

- You need gradients for optimization or inverse problems
- You want a Python-native workflow!
- You want GPU acceleration
- You want standalone optical simulations

Zemax / Commercial
---------------------

Commercial optical design softwares provide comprehensive tools for lens and
mirror system design. Zemax is commonly used in the design of telescopes.

**Advantages of commercial software:**

- Full-featured GUI for interactive design
- Extensive optimization algorithms built-in
- Handles complex optical systems (lenses, coatings, tolerancing)

**When to use IACTrace instead:**

- You don't want to pay money (duh)
- You want to integrate optical simulation with machine learning in jax
- You specifically need IACT features (hexagonal sensors, segmented primaries)
- You want to use open-source code where you can contribute features

ROBAST
---------------------

`ROBAST <https://robast.github.io/>`_ (ROOT-based simulator for ray tracing) is an 
open-source non-sequential ray-tracing library built on the ROOT framework, widely 
used within the CTA collaboration.

**Advantages of ROBAST:**

- Established in the CTAO community
- Handles complex segmented mirrors and aspherical lenses
- Integrates with ROOT analysis ecosystem
- Supports Winston cone / light concentrator simulations

**When to use IACTrace instead:**

- You need differentiable simulations for optimization
- You want GPU acceleration
- You prefer a pure Python workflow without ROOT dependencies

Summary Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15

   * - Feature
     - IACTrace
     - sim_telarray
     - Commercial
     - ROBAST
   * - Non-sequential ray tracing
     - No
     - No
     - Yes
     - Yes
   * - Complex 3D models
     - No
     - No
     - Yes
     - Yes
   * - Differentiable
     - Yes
     - No
     - Limited
     - No
   * - Python-native
     - Yes
     - No
     - No
     - No
   * - GPU support
     - Yes
     - No
     - Varies
     - No
   * - Open source
     - Yes
     - Partial
     - No
     - Yes
   * - Cost
     - Free
     - Free
     - $$$$
     - Free
