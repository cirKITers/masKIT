from typing import List

import pennylane as qml
import remote_cirq
from pennylane import numpy as np, GradientDescentOptimizer

from masked_parameters import MaskedParameters

np.random.seed(1337)


class ExtendedGradientDescentOptimizer(GradientDescentOptimizer):
    def step_cost_and_grad(self, objective_fn, *args, grad_fn=None, **kwargs):
        """
        This function copies the functionality of the GradientDescentOptimizer
        one-to-one but changes the return statement to also return the gradient.
        """
        gradient, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(gradient, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward, gradient
        return new_args, forward, gradient


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


def train_circuit(wires=5, layers=5, steps=500, sim_local=True, use_dropout=False):
    def ensemble_step(branches: List[MaskedParameters], optimizer, *args, step_count=1):
        """
        Targeting 26-32 Qubits on Floq is possible, so for the ensemble we might just
        roll out the ensembles in the available range of Floq.
        """
        branch_costs = []
        gradients = []
        for branch in branches:
            start_params = branch.params
            for _ in range(step_count):
                end_params, cost, gradient = optimizer.step_cost_and_grad(
                    *args, start_params, mask=branch.mask)
            branch.params = end_params
            branch_costs.append(cost)
            gradients.append(gradient)
        minimum_cost = min(branch_costs)
        minimum_index = branch_costs.index(minimum_cost)
        return branches[minimum_index], branch_costs[minimum_index], gradients[minimum_index]

    dev = get_device(sim_local, wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers*wires)]

    masked_params = MaskedParameters(np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))

    opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    # opt = qml.AdamOptimizer(stepsize=0.01)

    circuit = qml.QNode(variational_circuit, dev)

    step_count = steps // 2 // 3 if use_dropout else steps // 2
    for step in range(step_count):
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

        best_branch, minimum_cost, gradient = ensemble_step(
            branches,
            opt,
            lambda params, mask=None: cost(
                circuit, params, wires, layers, rotations, mask),
            step_count=2)

        masked_params = best_branch
        current_cost = minimum_cost

        print("Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(step, current_cost, np.var(gradient)))

    print(masked_params.params)
    print(masked_params.mask)


if __name__ == "__main__":
    train_circuit(sim_local=True, use_dropout=False, steps=20)
    # example values for layers = 15
    # steps | dropout = False | dropout = True
    # 20:     0.99600,          0.90200
    # 50:     0.98900,          0.92500
    # steps are normalised with regard to # branches
