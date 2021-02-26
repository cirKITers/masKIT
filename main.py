from typing import List

import pennylane as qml
import remote_cirq
from pennylane import numpy as np, GradientDescentOptimizer, AdamOptimizer
from collections import deque

from masked_parameters import MaskedParameters, PerturbationMode, PerturbationAxis
from iris import load_iris, cross_entropy

from log_results import log_results

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self._stepsize})"


class ExtendedAdamOptimizer(ExtendedGradientDescentOptimizer, AdamOptimizer):
    pass


def get_device(sim_local, wires, analytic=True):
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


def circuit_iris(params, data, wires, layers, rotations, dropouts):
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


def cost(circuit, params, wires, layers, rotations, dropouts):
    return 1 - circuit(params, wires, layers, rotations, dropouts)[0]


def cost_iris(circuit, params, data, target, wires, layers, rotations, dropouts):
    prediction = circuit(params, data, wires, layers, rotations, dropouts)
    return cross_entropy(predictions=prediction, targets=target)


def train_iris(wires=5, layers=5, starting_layers=5, epochs=5, sim_local=True, use_dropout=False, use_classical_dropout=False, testing=True):
    dev = get_device(sim_local, wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers*wires)]

    current_layers = layers if use_dropout else starting_layers
    params_uniform = np.random.uniform(low=-np.pi, high=np.pi, size=(current_layers, wires))
    params_zero = np.zeros((layers-current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    masked_params = MaskedParameters(params_combined)
    # masked_params = MaskedParameters(np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))
    
    # opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    opt = ExtendedAdamOptimizer(stepsize=0.01)

    circuit = qml.QNode(circuit_iris, dev)
    x_train, y_train, x_test, y_test = load_iris()

    for epoch in range(epochs):
        for step, (data, target) in enumerate(zip(x_train, y_train)):
            if use_dropout:
                center_params = masked_params
                left_branch_params = masked_params.copy()
                right_branch_params = masked_params.copy()
                # perturb the right params
                left_branch_params.perturb(1, mode=PerturbationMode.REMOVE)
                right_branch_params.perturb()
                branches = [center_params, left_branch_params, right_branch_params]
            elif use_classical_dropout:
                masked_params.reset()
                masked_params.perturb(masked_params.params.size // 10, mode=PerturbationMode.ADD)
                branches = [masked_params]
            else:
                branches = [masked_params]

            best_branch, minimum_cost, gradient = ensemble_step(
                branches,
                opt,
                lambda params, mask=None: cost_iris(
                    circuit, params, data, target, wires, current_layers, rotations, mask),
                step_count=2)

            masked_params = best_branch
            current_cost = minimum_cost
            # get the real gradients as gradients also contain values from dropped gates
            real_gradients = masked_params.apply_mask(gradient)

            print("Epoch: {:2d} | Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(epoch, step, current_cost, np.var(real_gradients[0:current_layers])))

        if testing:
            correct = 0
            N = len(x_test)
            costs = []
            for step, (data, target) in enumerate(zip(x_test, y_test)):
                test_mask = np.zeros_like(masked_params.params, dtype=bool, requires_grad=False) 
                output = circuit(masked_params.params, data, wires, current_layers, rotations, test_mask)
                c = cost_iris(circuit, masked_params.params, data, target, wires, current_layers, rotations, test_mask)
                costs.append(c)
                same = np.argmax(target) == np.argmax(output)
                if same:
                    correct += 1
                print("Label: {} Output: {} Correct: {}".format(target, output, same))
            print("Accuracy = {} / {} = {} \nAvg Cost: {}".format(correct, N, correct/N, np.average(costs)))

        if epoch % 2 == 0 and epoch > 0 and current_layers < layers:
            current_layers += 1
            print(f"Increased number of layers from {current_layers-1} to {current_layers}")

    print(masked_params.params)
    print(masked_params.mask)


def ensemble_step(branches: List[MaskedParameters], optimizer, *args, step_count=1):
    """
    Targeting 26-32 Qubits on Floq is possible, so for the ensemble we might just
    roll out the ensembles in the available range of Floq.
    """
    branch_costs = []
    gradients = []
    for branch in branches:
        params = branch.params
        for _ in range(step_count):
            params, cost, gradient = optimizer.step_cost_and_grad(
                *args, params, mask=branch.mask)
        branch.params = params
        branch_costs.append(args[0](params, mask=branch.mask))
        gradients.append(gradient)
    minimum_cost = min(branch_costs)
    minimum_index = branch_costs.index(minimum_cost)
    return branches[minimum_index], branch_costs[minimum_index], gradients[minimum_index]


def train_test_iris(wires=5, layers=5, sim_local=True, percentage=0.05, epsilon=0.01, testing=True):
    dev = get_device(sim_local=sim_local, wires=wires)
    circuit = qml.QNode(circuit_iris, dev)
    x_train, y_train, x_test, y_test = load_iris()

    # opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    opt = ExtendedAdamOptimizer(stepsize=0.01)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    cost_fn = lambda params, mask=None: cost_iris(
                    circuit, params, data, target, wires, layers, rotations, mask)

    amount = int(wires * layers * percentage)
    masked_params = MaskedParameters(
        np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))
    masked_params.perturbation_axis = PerturbationAxis.RANDOM
    masked_params.perturb(int(layers * wires * 0.5))

    costs = deque(maxlen=5)
    perturb = False
    for step, (data, target) in enumerate(zip(x_train, y_train)):
        if perturb:
            left_branch = masked_params.copy()
            left_branch.perturb(amount=1, mode=PerturbationMode.ADD)
            right_branch = masked_params.copy()
            right_branch.perturb(amount=amount, mode=PerturbationMode.REMOVE)
            branches = [masked_params, left_branch, right_branch]
            perturb = False
        else:
            branches = [masked_params]
        branch, current_cost, gradient = ensemble_step(branches, opt, cost_fn)
        masked_params = branch
        costs.append(current_cost)
        print("Step: {:4d} | Cost: {: .5f}".format(step, current_cost))
        if len(costs) >= 5 and current_cost > .1:
            if sum([abs(cost - costs[index + 1]) for index, cost in enumerate(list(costs)[:-1])]) < epsilon:
                print("======== allowing to perturb =========")
                if np.sum(masked_params.mask) >= layers * wires * .3:
                    masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                elif current_cost < 0.25 and np.sum(masked_params.mask) >= layers * wires * .05:
                    masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                costs.clear()
                perturb = True

    if testing:
        correct = 0
        N = len(x_test)
        costs = []
        for step, (data, target) in enumerate(zip(x_test, y_test)):
            output = circuit(masked_params.params, data, wires, layers,
                             rotations, masked_params.mask)
            c = cost_iris(circuit, masked_params.params, data, target, wires,
                          layers, rotations, masked_params.mask)
            costs.append(c)
            same = np.argmax(target) == np.argmax(output)
            if same:
                correct += 1
            print("Label: {} Output: {} Correct: {}".format(target, output, same))
        print("Accuracy = {} / {} = {} \nAvg Cost: {}".format(correct, N, correct / N,
                                                              np.average(costs)))

    print(masked_params.mask)
    print(masked_params.params)


@log_results
def train_test(optimiser, wires=5, layers=5, sim_local=True, steps=100, percentage=0.05, epsilon=0.01, cost_span: int=5, log_interval: int=5, use_dropout=True, seed=1337):
    np.random.seed(seed)

    logging_costs = {}
    logging_branches = {}
    logging_branch_selection = {}
    logging_branch_enforcement = {}
    logging_gate_count = {}
    logging_cost_values = []
    logging_gate_count_values = []

    dev = get_device(sim_local=sim_local, wires=wires)
    circuit = qml.QNode(variational_circuit, dev)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    cost_fn = lambda params, mask=None: cost(circuit, params, wires, layers, rotations, mask)

    amount = int(wires * layers * percentage)
    masked_params = MaskedParameters(
        np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))
    masked_params.perturbation_axis = PerturbationAxis.RANDOM

    costs = deque(maxlen=cost_span)
    perturb = False
    for step in range(steps):
        if use_dropout:
            if perturb:
                left_branch = masked_params.copy()
                left_branch.perturb(amount=1, mode=PerturbationMode.ADD)
                right_branch = masked_params.copy()
                right_branch.perturb(amount=amount, mode=PerturbationMode.REMOVE)
                branches = [masked_params, left_branch, right_branch]
                perturb = False
                logging_branches[step] = {
                    "center": "No perturbation",
                    "left": {"amount": 1, "mode": PerturbationMode.ADD, "axis": left_branch.perturbation_axis},
                    "right": {"amount": amount, "mode": PerturbationMode.REMOVE, "axis": right_branch.perturbation_axis}
                }
            else:
                branches = [masked_params]
        else:
            branches = [masked_params]
        branch, current_cost, gradient = ensemble_step(branches, optimiser, cost_fn)
        branch_index = branches.index(branch)
        logging_branch_selection[step] = "center" if branch_index == 0 else "left" if branch_index == 1 else "right"
        masked_params = branch
        # print("Step: {:4d} | Cost: {: .5f}".format(step, current_cost))

        logging_cost_values.append(current_cost.unwrap())
        logging_gate_count_values.append(np.sum(masked_params.mask))
        if step % log_interval == 0:
            # perform logging
            logging_costs[step] = np.average(logging_cost_values)
            logging_gate_count[step] = np.average(logging_gate_count_values)
            logging_cost_values.clear()
            logging_gate_count_values.clear()

        if use_dropout:
            costs.append(current_cost)
            if len(costs) >= cost_span and current_cost > .1:
                if sum([abs(cost - costs[index + 1]) for index, cost in enumerate(list(costs)[:-1])]) < epsilon:
                    # print("======== allowing to perturb =========")
                    if np.sum(masked_params.mask) >= layers * wires * .3:
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                        logging_branch_enforcement[step + 1] = {
                            "amount": 1,
                            "mode": PerturbationMode.REMOVE,
                            "axis": masked_params.perturbation_axis}
                    elif current_cost < 0.25 and np.sum(masked_params.mask) >= layers * wires * .05:
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                        logging_branch_enforcement[step + 1] = {
                            "amount": 1,
                            "mode": PerturbationMode.REMOVE,
                            "axis": masked_params.perturbation_axis}
                    costs.clear()
                    perturb = True
    return {
        "costs": logging_costs,
        "final_cost": current_cost.unwrap(),
        "branch_enforcements": logging_branch_enforcement,
        "dropouts": logging_gate_count,
        "branches": logging_branches,
        "branch_selections": logging_branch_selection,
        "params": masked_params.params.unwrap(),
        "mask": masked_params.mask.unwrap()
    }


def train_circuit(wires=5, layers=5, starting_layers=5, steps=500, sim_local=True, use_dropout=False):
    dev = get_device(sim_local, wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers*wires)]

    current_layers = layers if use_dropout else starting_layers
    params_uniform = np.random.uniform(low=-np.pi, high=np.pi, size=(current_layers, wires))
    params_zero = np.zeros((layers-current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    masked_params = MaskedParameters(params_combined)
    # masked_params = MaskedParameters(np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires)))
    
    # opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    opt = ExtendedAdamOptimizer(stepsize=0.01)

    circuit = qml.QNode(variational_circuit, dev)

    step_count = steps // 2 // 3 if use_dropout else steps // 2
    for step in range(step_count):
        if use_dropout:
            center_params = masked_params
            left_branch_params = masked_params.copy()
            right_branch_params = masked_params.copy()
            # perturb the right params
            left_branch_params.perturb(1, mode=PerturbationMode.REMOVE)
            right_branch_params.perturb()
            branches = [center_params, left_branch_params, right_branch_params]
        else:
            branches = [masked_params]

        best_branch, minimum_cost, gradient = ensemble_step(
            branches,
            opt,
            lambda params, mask=None: cost(
                circuit, params, wires, current_layers, rotations, mask),
            step_count=2)

        masked_params = best_branch
        current_cost = minimum_cost
        # get the real gradients as gradients also contain values from dropped gates
        real_gradients = masked_params.apply_mask(gradient)

        print("Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(step, current_cost, np.var(real_gradients[0:current_layers])))

        if step % 40 == 0 and step > 0 and current_layers < layers:
            current_layers += 1
            print(f"Increased number of layers from {current_layers-1} to {current_layers}")

    # print(masked_params.params)
    print(masked_params.mask)


if __name__ == "__main__":
    opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    # opt = ExtendedAdamOptimizer(stepsize=0.01)

    # train_iris(sim_local=True, use_dropout=False, epochs=3)
    # train_circuit(sim_local=True, use_dropout=False, steps=200, wires=10)
    print(train_test(opt, steps=10, wires=5, sim_local=True))
    # train_test_iris(wires=20, layers=10, sim_local=True, percentage=0.01)
    # example values for layers = 15
    # steps | dropout = False | dropout = True
    # 20:     0.99600,          0.91500
    # 50:     0.98900,          0.81700
    # steps are normalised with regard to # branches
