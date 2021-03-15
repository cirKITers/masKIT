import pennylane as qml
from pennylane import numpy as np


def basic_variational_circuit(params, wires, layers, rotations, dropouts):
    for wire in range(wires):
        qml.RY(np.pi / 4, wires=wire)
    r = -1
    for layer in range(layers):
        for wire in range(wires):
            r += 1
            if dropouts[layer][wire] is True:
                continue
            if rotations[r] == 0:
                rotation = qml.RX
            elif rotations[r] == 1:
                rotation = qml.RY
            else:
                rotation = qml.RZ
            rotation(params[layer][wire], wires=wire)

        for wire in range(0, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])


def variational_circuit(params, wires, layers, rotations, dropouts):
    basic_variational_circuit(
        params=params,
        wires=wires,
        layers=layers,
        rotations=rotations,
        dropouts=dropouts,
    )
    return qml.probs(wires=range(wires))


def iris_circuit(params, data, wires, layers, rotations, dropouts):
    qml.templates.embeddings.AngleEmbedding(features=data, wires=range(4), rotation="X")
    basic_variational_circuit(
        params=params,
        wires=wires,
        layers=layers,
        rotations=rotations,
        dropouts=dropouts,
    )
    return qml.probs(wires=[0, 1])
