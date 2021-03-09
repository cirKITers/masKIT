import pennylane as qml
from pennylane import numpy as np

def variational_circuit(params, wires, layers, rotations, dropouts):
    for w in range(wires):
        qml.RY(np.pi/4, wires=w) 
    r = -1
    for l in range(layers):
        for w in range(wires):
            r += 1
            if dropouts[l][w] == True:
                continue
            if rotations[r] == 0:
                rotation = qml.RX
            elif rotations[r] == 1:
                rotation = qml.RY
            else:
                rotation = qml.RZ
            rotation(params[l][w], wires=w)
    
        for w in range(0, wires - 1, 2):
            qml.CZ(wires=[w, w + 1])
        for w in range(1, wires - 1, 2):
            qml.CZ(wires=[w, w + 1])
    return qml.probs(wires=range(wires))


def iris_circuit(params, data, wires, layers, rotations, dropouts):
    qml.templates.embeddings.AngleEmbedding(features=data, wires=range(4), rotation="X")

    for w in range(wires):
        qml.RY(np.pi/4, wires=w) 
    r = -1
    for l in range(layers):
        for w in range(wires):
            r += 1
            if dropouts[l][w] == True:
                continue
            if rotations[r] == 0:
                rotation = qml.RX
            elif rotations[r] == 1:
                rotation = qml.RY
            else:
                rotation = qml.RZ
            rotation(params[l][w], wires=w)
    
        for w in range(0, wires - 1, 2):
            qml.CZ(wires=[w, w + 1])
        for w in range(1, wires - 1, 2):
            qml.CZ(wires=[w, w + 1])
    return qml.probs(wires=[0,1])