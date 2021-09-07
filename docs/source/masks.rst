================
Masking Circuits
================

.. module:: maskit.masks
    :synopsis: Implementation of Masks, MaskedCircuits and FreezableMaskedCircuits

The concept of masking is fairly simple: When a position in a mask is set to True,
the object referenced will not appear when the mask is applied. In our case 
referenced objects are gates in a quantum circuit that can be removed by applying masks.

When considering different ansätze from literature, e.g. when creating parameterised
quantum circuits that often rely on repeating layers, it becomes clear that a single
mask cannot be easily used while still maintaining comprehensability, usability, and 
maintainability. We use a composition of masks instead to create the concept of a 
:py:class:`~.MaskedCircuit`. 

A :py:class:`~.MaskedCircuit` has a concept of *wires*, *layers* as well as
*entangling gates* that make up the variational part of the circuit.
Each of them is represented by a separate :py:class:`~.Mask`. The final mask that is 
applied to the underlying parameters is the intersection of all masks.

You can change and adapt masks by different methods:

- you can manually set specific positions in a mask,
- you can perturb a mask on a random position, and
- you can shrink a mask.

.. autoclass:: PerturbationAxis
    :members:

.. autoclass:: PerturbationMode
    :members:

.. autoclass:: MaskedCircuit

.. automethod:: MaskedCircuit.perturb

.. automethod:: MaskedCircuit.shrink
