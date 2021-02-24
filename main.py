import pennylane as qml
import remote_cirq
from random import choice, randrange
from pennylane import numpy as np

from masked_parameters import MaskedParameters

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


def cost(circuit, params, wires, layers, rotations, dropouts):
    return 1 - circuit(params, wires, layers, rotations, dropouts)[0]


def determine_dropout(params, dropout, epsilon=0.01, factor=0.2, difference=0):
    """
    Determines new dropout based on previous dropout and current parameters.

    Args:
        params ([type]): Parameters for the current training
        dropout ([type]): Dropout currently in use

    Returns:
        [type]: Dropout for next training step
    """
    new_dropout = dropout.copy()
    indices = np.argwhere((params <= epsilon) & (params >= -epsilon))
    if len(indices) > 0:
        index = choice(indices)
        new_dropout[index[0], index[1]] = 1
    if difference > epsilon:
        i_dim = len(params)
        j_dim = len(params[0])
        max_count = i_dim * j_dim
        if np.sum(new_dropout) / max_count < factor:
            # in case we have little dropouts, add one
            rand_i = randrange(0, i_dim)
            rand_j = randrange(0, j_dim)
            new_dropout[rand_i, rand_j] = 1
        elif np.sum(new_dropout) / max_count > factor:
            # in case the dropout is higher than factor, we should randomly remove one
            current_indices = np.argwhere(new_dropout == 1)
            index = choice(current_indices)
            new_dropout[index[0], index[1]] = 0
    return new_dropout


def train_circuit(wires=5, layers=5, steps=500, sim_local=True, use_dropout=False):
    dev = get_device(sim_local, wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers*wires)]

    masked_params = MaskedParameters(np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))

    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    last_cost = None

    circuit = qml.QNode(variational_circuit, dev)

    # grad = qml.grad(circuit, argnum=0)
    # gradient = grad(params, wires=wires, layers=layers, rotations=rotations, dropouts=dropouts)

    step_count = steps // 2 // 3 if use_dropout else steps // 2
    for step in range(step_count):
        current_cost = cost(circuit, masked_params.params, wires, layers, rotations, masked_params.mask)

        if use_dropout:
            center_params = masked_params
            left_branch_params = masked_params.copy()
            right_branch_params = masked_params.copy()
            # perturb the right params
            left_branch_params.perturb(-1)
            right_branch_params.perturb()
            branches = [center_params, left_branch_params, right_branch_params]
        else:
            branches = [masked_params]

        branch_costs = []
        for params in branches:
            for _ in range(2):
                params.params = opt.step(
                    lambda p: cost(circuit, p, wires, layers, rotations, params.mask),
                    params.params)
            branch_costs.append(cost(circuit, params.params, wires, layers, rotations, params.mask))

        minimum_cost = min(branch_costs)
        index = branch_costs.index(minimum_cost)
        print(f"picked index {index}")
        masked_params = branches[index]
        current_cost = branch_costs[index]

        # print("Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(step, new_cost, np.var(gradient)))
        print("Step: {:4d} | Cost: {: .5f}".format(step, current_cost))
    print(masked_params.params)
    print(masked_params.mask)


if __name__ == "__main__":
    train_circuit(sim_local=True, use_dropout=False, steps=20)
    # example values for layers = 15
    # steps | dropout = False | dropout = True
    # 20:     0.99600,          0.90200
    # 50:     0.98900,          0.92500
    # steps are normalised with regard to # branches
