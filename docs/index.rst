.. masKIT documentation master file, created by
   sphinx-quickstart on Wed Sep  1 16:46:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ensemble-based gate dropouts for quantum circuits
=====================================================

.. toctree::
   :maxdepth: 2
   :caption: Usage and Guides
   :hidden:

   source/getting_started
   source/masks
   source/ensembles
   source/glossary
   Module Index <source/api/modules>

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   source/changelog
   CONTRIBUTING.md
   source/publishing

MasKIT is a framework that provides masking functionality in the context of
parameterized quantum circuits for PennyLane.
It simplifies researching trainability and expressivity of those circuits by
enabling to dynamically mask gates within a circuit.
The framework is designed to act as a drop-in replacement and therefore allows
to enhance your existing PennyLane projects with low effort.

The masking is supported on different axes, i.e. layers, wires, parameters, and
entangling gates, for different modes, i.e. adding, removing, inverting.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
