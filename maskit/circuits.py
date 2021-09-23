import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple

from maskit._masks import PerturbationAxis as Axis, DropoutMask
from maskit._masked_circuits import MaskedCircuit
from maskit.utils import cross_entropy


def cost(
    circuit,
    params,
    rotations: List,
    masked_circuit: MaskedCircuit,
):
    return 1 - circuit(params, rotations, masked_circuit)[0]


def cost_basis(
    circuit,
    params,
    data,
    target,
    rotations: List,
    masked_circuit: MaskedCircuit,
    wires: int,
    wires_to_measure: Tuple[int, ...],
):
    prediction = circuit(
        params, data, rotations, masked_circuit, wires, wires_to_measure
    )
    return cross_entropy(predictions=prediction, targets=target)


def basic_variational_circuit(params, rotations, masked_circuit: MaskedCircuit):
    full_parameters = masked_circuit.expanded_parameters(params)
    wires = masked_circuit.wires
    dropout_mask = masked_circuit.full_mask(DropoutMask)
    for wire, _is_masked in enumerate(masked_circuit.mask(Axis.WIRES)):
        qml.RY(np.pi / 4, wires=wire)
    r = -1
    for layer, _is_layer_masked in enumerate(masked_circuit.mask(Axis.LAYERS)):
        for wire, _is_wire_masked in enumerate(masked_circuit.mask(Axis.WIRES)):
            r += 1
            if dropout_mask[layer][wire]:
                continue
            if rotations[r] == 0:
                rotation = qml.RX
            elif rotations[r] == 1:
                rotation = qml.RY
            else:
                rotation = qml.RZ
            rotation(full_parameters[layer][wire], wires=wire)

        for wire in range(0, wires - 1, 2):
            if (
                Axis.ENTANGLING in masked_circuit.masks
                and masked_circuit.mask(Axis.ENTANGLING)[layer, wire]
            ):
                continue
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            if (
                Axis.ENTANGLING in masked_circuit.masks
                and masked_circuit.mask(Axis.ENTANGLING)[layer, wire]
            ):
                continue
            qml.CZ(wires=[wire, wire + 1])


def variational_circuit(params, rotations, masked_circuit):
    masked_circuit = masked_circuit.unwrap()
    basic_variational_circuit(
        params=params,
        rotations=rotations,
        masked_circuit=masked_circuit,
    )
    return qml.probs(
        wires=range(len(masked_circuit.mask(axis=Axis.WIRES, mask_type=DropoutMask)))
    )


def basis_circuit(params, data, rotations, masked_circuit, wires, wires_to_measure):
    masked_circuit = masked_circuit.unwrap()
    qml.templates.embeddings.AngleEmbedding(
        features=data, wires=range(wires), rotation="X"
    )
    basic_variational_circuit(
        params=params, rotations=rotations, masked_circuit=masked_circuit
    )
    return qml.probs(wires=wires_to_measure.tolist())
