Example Gallery
===============

This section provides worked examples demonstrating IACTrace capabilities.
Each example is a Jupyter notebook that you can download and run yourself.

----

Basic Examples
--------------

Simple optical systems and fundamental concepts. Start here if you're new
to IACTrace.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Parabolic Telescope
      :link: Parabolic
      :link-type: doc

      A simple single mirror parabolic telescope.

   .. grid-item-card:: Cassegrain Telescope
      :link: Cassegrain
      :link-type: doc

      Two-mirror Cassegrain optical system.

----

H.E.S.S. Telescopes
-------------------

Simulations of the H.E.S.S. telescope array in Namibia, including the Phase I 
telescopes and the Phase II telescope.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: H.E.S.S. Phase I (CT3)
      :link: HESS_I
      :link-type: doc

      Simulation of the H.E.S.S. I telescope
      
   .. grid-item-card:: H.E.S.S. Phase II (CT5)
      :link: HESS_II
      :link-type: doc

      Simulation of the H.E.S.S. II telescope

----

CTAO Telescopes
-------------------

Simulations of CTAO-like telescopes (obstruction geometry gathered from images)

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: MST
      :link: MST
      :link-type: doc

      Simulation of the MST
      
   .. grid-item-card:: LST
      :link: LST
      :link-type: doc

      Simulation of the LST

----

Others
------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Response Matrix
      :link: ResponseMatrix
      :link-type: doc

      Computing per-source per-pixel effective aperture in a single
      render pass via :meth:`iactrace.Camera.response_matrix`.

.. toctree::
   :maxdepth: 1
   :hidden:

   Parabolic
   Cassegrain
   HESS_I
   HESS_II
   MST
   LST
   ResponseMatrix