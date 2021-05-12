from maskit.masks import MaskedCircuit
import pennylane as qml
from pennylane import numpy as np


def basic_variational_circuit(params, rotations, masked_circuit: MaskedCircuit):
    full_parameters = masked_circuit.expanded_parameters(params)
    wires = len(masked_circuit.wire_mask)
    mask = masked_circuit.mask
    for wire, _is_masked in enumerate(masked_circuit.wire_mask):
        qml.RY(np.pi / 4, wires=wire)
    r = -1
    for layer, _is_layer_masked in enumerate(masked_circuit.layer_mask):
        for wire, _is_wire_masked in enumerate(masked_circuit.wire_mask):
            r += 1
            if mask[layer][wire]:
                continue
            if rotations[r] == 0:
                rotation = qml.RX
            elif rotations[r] == 1:
                rotation = qml.RY
            else:
                rotation = qml.RZ
            rotation(full_parameters[layer][wire], wires=wire)

        for wire in range(0, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])


def variational_circuit(params, rotations, masked_circuit):
    masked_circuit = masked_circuit.unwrap()
    basic_variational_circuit(
        params=params,
        rotations=rotations,
        masked_circuit=masked_circuit,
    )
    return qml.probs(wires=range(len(masked_circuit.wire_mask)))


def iris_circuit(params, data, rotations, masked_circuit):
    masked_circuit = masked_circuit.unwrap()
    qml.templates.embeddings.AngleEmbedding(features=data, wires=range(4), rotation="X")
    basic_variational_circuit(
        params=params, rotations=rotations, masked_circuit=masked_circuit
    )
    return qml.probs(wires=[0, 1])
