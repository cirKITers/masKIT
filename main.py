import pennylane as qml
import remote_cirq
from random import choice, randrange
from pennylane import numpy as np

np.random.seed(1337)

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

        H = np.zeros((2 ** wires, 2 ** wires))
        H[0, 0] = 1
        wirelist = [i for i in range(wires)]
        return qml.expval(qml.Hermitian(H, wirelist))
    return variational_circuit

def cost(dev, wires, layers, params, rotations, dropouts):
    return my_circuit(dev, wires, layers, params, rotations, dropouts)()

def determine_dropout(params, dropout, epsilon=0.01, factor=0.2, difference=0):
    """
    Determines new dropout based on previous dropout and current parameters.

    Args:
        params ([type]): Parameters for the current training
        dropout ([type]): Dropout currently in use

    Returns:
        [type]: Dropout for next training step
    """
    new_dropout = np.zeros_like(params)
    i_dim = len(params)
    j_dim = len(params[0])
    for i in range(i_dim):
        for j in range(j_dim):
            if 0 - epsilon <= params[i][j] >= 0 + epsilon:
                new_dropout[i][j] = 1
    max_count = i_dim * j_dim
    if difference > epsilon:
        while np.sum(new_dropout) / max_count <= factor:
            rand_i = randrange(0, i_dim)
            rand_j = randrange(0, j_dim)
            new_dropout[rand_i][rand_j] = 1
    return new_dropout

def train_circuit(wires=5, layers=5, steps=500, sim_local=True):
    dev = get_device(sim_local, wires=wires)
    #circuit = qml.QNode(variational_circuit, dev)

    rotation_choices = [0, 1, 2]
    rotations = []
    for _ in range(layers*wires):
        rotation = np.random.choice(rotation_choices)
        rotations.append(rotation)

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires))
    dropouts = determine_dropout(params, np.zeros_like(params))

    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    last_cost = None

    for step in range(steps):
        new_cost = cost(dev, wires, layers, params, rotations, dropouts)
        if last_cost is None:
            last_cost = new_cost
        difference = last_cost - new_cost
        
        print(step, new_cost)
        params = opt.step(lambda p: cost(dev, wires, layers, p, rotations, dropouts), params)
        # determine new dropouts based on new parameter values
        dropouts = determine_dropout(params, dropouts, difference=difference)
        last_cost = new_cost


if __name__ == "__main__":
    train_circuit()