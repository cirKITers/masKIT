import pennylane as qml
import remote_cirq
from random import choice
from pennylane import numpy as np

np.random.seed(1447)

def get_device(sim_local, wires, analytic=False):
    if sim_local:
        dev = qml.device("default.qubit", 
                         wires=wires, 
                         analytic=analytic)
    else:
        print("Asking Google...")
        with open('key.txt') as f:
            api_key = f.readline()[:-1]
        sim = remote_cirq.RemoteSimulator(api_key)
        dev = qml.device("cirq.simulator",
                         wires=wires,
                         simulator=sim,
                         analytic=analytic)
    return dev

def my_circuit(dev, wires, layers, params, rotations, dropouts):
    @qml.qnode(dev)
    def variational_circuit():
        for w in range(wires):
            qml.RY(np.pi/4, wires=w) 
        r = 0
        for l in range(layers):
            for w in range(wires):
                if dropouts[l][w] == 1:
                    r += 1
                    continue
                if rotations[r] == 0:
                    rotation = qml.RX
                elif rotations[r] == 1:
                    rotation = qml.RY
                else:
                    rotation = qml.RZ
                r += 1
                rotation(params[l][w], wires=w)
        
            for w in range(wires-1):
                qml.CZ(wires=[w, w+1])

        return qml.probs(0)
    return variational_circuit

def cost(dev, wires, layers, params, rotations, dropouts):
    return 1 - my_circuit(dev, wires, layers, params, rotations, dropouts)()[0]

def train_circuit(wires=5, layers=5, steps=500, sim_local=True):
    dev = get_device(sim_local, wires=wires)
    #circuit = qml.QNode(variational_circuit, dev)

    rotation_choices = [0, 1, 2]
    rotations = []
    for _ in range(layers*wires):
        rotation = choice(rotation_choices)
        rotations.append(rotation)

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires))
    dropouts = np.zeros_like(params)

    opt = qml.GradientDescentOptimizer(stepsize=0.01)

    for s in range(steps):
        c = cost(dev, wires, layers, params, rotations, dropouts)
        print(s, c)
        params = opt.step(lambda p: cost(dev, wires, layers, p, rotations, dropouts), params)


if __name__ == "__main__":
    train_circuit()