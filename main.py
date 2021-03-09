from typing import List

import pennylane as qml
import remote_cirq
from pennylane import numpy as np, GradientDescentOptimizer, AdamOptimizer
from collections import deque

from masked_parameters import MaskedParameters, PerturbationMode, PerturbationAxis
from iris import load_iris
from utils import cross_entropy, check_params
from circuits import variational_circuit, iris_circuit
from log_results import log_results
from optimizers import ExtendedAdamOptimizer, ExtendedGradientDescentOptimizer

#np.random.seed(1337)


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


def cost(circuit, params, wires, layers, rotations, dropouts):
    return 1 - circuit(params, wires, layers, rotations, dropouts)[0]


def cost_iris(circuit, params, data, target, wires, layers, rotations, dropouts):
    prediction = circuit(params, data, wires, layers, rotations, dropouts)
    return cross_entropy(predictions=prediction, targets=target)


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


def train_test_iris(wires=5, layers=5, sim_local=True, percentage=0.05, epsilon=0.01, testing=True, seed=1337):
    np.random.seed(seed)
    dev = get_device(sim_local=sim_local, wires=wires)
    circuit = qml.QNode(iris_circuit, dev)
    x_train, y_train, x_test, y_test = load_iris()

    opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    # opt = ExtendedAdamOptimizer(stepsize=0.01)

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


def train(train_params):
    np.random.seed(train_params["seed"])

    # set up circuit, training, dataset  
    wires = train_params["wires"]
    layers = train_params["layers"]
    dev = get_device(train_params["sim_local"], wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers*wires)]

    current_layers = layers if train_params["dropout"] is not ["growing"] else train_params["starting_layers"]
    
    if train_params["optimizer"] == "gd":
        opt = ExtendedGradientDescentOptimizer(train_params["step_size"])
    elif train_params["optimizer"] == "adam":
        opt = ExtendedAdamOptimizer(train_params["step_size"])
    # TODO
    # steps = train_params["steps"] // 2 // 3 if train_params["use_dropout"] else train_params["steps"] // 2
    # do we need this? Maybe if elif for every possiple "dropout"
    steps = train_params["steps"]

    if train_params["dataset"] == "simple":
        circuit = qml.QNode(variational_circuit, dev)
        cost_fn = lambda params, mask=None: cost(
            circuit, params, wires, current_layers, rotations, mask)
                
    elif train_params["dataset"] == "iris":
        circuit = qml.QNode(iris_circuit, dev)
        x_train, y_train, x_test, y_test = load_iris()
        cost_fn = lambda params, mask=None: cost_iris(
            circuit, params, data, target, wires, current_layers, rotations, mask)
                

    # set up parameters
    params_uniform = np.random.uniform(low=-np.pi, high=np.pi, size=(current_layers, wires))
    params_zero = np.zeros((layers-current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    masked_params = MaskedParameters(params_combined)

    if train_params["dropout"] == "eileen":
        masked_params.perturb(int(layers * wires * 0.5))
        amount = int(wires * layers * train_params["percentage"])
        perturb = False
        costs = deque(maxlen=5)

    # -----------------------------
    # ======= TRAINING LOOP =======
    # -----------------------------
    for step in range(steps):
        if train_params["dropout"] == "random":
            center_params = masked_params
            left_branch_params = masked_params.copy()
            right_branch_params = masked_params.copy()
            # perturb the right params
            left_branch_params.perturb(1, mode=PerturbationMode.REMOVE)
            right_branch_params.perturb()
            branches = [center_params, left_branch_params, right_branch_params]
        elif train_params["dropout"] == "classical":
            masked_params.reset()
            masked_params.perturb(masked_params.params.size // 10, mode=PerturbationMode.ADD)
            branches = [masked_params]
        elif train_params["dropout"] == "growing":
            # TODO useful condition
            # maybe combine with other dropouts
            if step > 0 and step % 1000 == 0:
                current_layers += 1
            branches = [masked_params] 
        elif train_params["dropout"] == "eileen":
            if perturb:
                left_branch = masked_params.copy()
                left_branch.perturb(amount=1, mode=PerturbationMode.ADD)
                right_branch = masked_params.copy()
                right_branch.perturb(amount=amount, mode=PerturbationMode.REMOVE)
                branches = [masked_params, left_branch, right_branch]
                perturb = False
            else:
                branches = [masked_params]
        else:
            branches = [masked_params]

        if train_params["dataset"] == "iris":
            data = x_train[step % len(x_train)]
            target =  y_train[step % len(y_train)]
   
        masked_params, current_cost, gradient = ensemble_step(branches, opt, cost_fn)
   
        # get the real gradients as gradients also contain values from dropped gates
        real_gradients = masked_params.apply_mask(gradient)

        print("Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(step, current_cost, np.var(real_gradients[0:current_layers])))

        if train_params["dropout"] == "eileen":
            if len(costs) >= train_params["cost_span"] and current_cost > .1:
                if sum([abs(cost - costs[index + 1]) for index, cost in enumerate(list(costs)[:-1])]) < train_params["epsilon"]:
                    print("======== allowing to perturb =========")
                    if np.sum(masked_params.mask) >= layers * wires * .3:
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                    elif current_cost < 0.25 and np.sum(masked_params.mask) >= layers * wires * .05:
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                    costs.clear()
                    perturb = True

    if train_params["testing"]:
        if train_params["dataset"] == "simple":
            pass
        elif train_params["dataset"] == "iris":
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


    print(masked_params.params)
    print(masked_params.mask)


if __name__ == "__main__":
    opt = ExtendedGradientDescentOptimizer(stepsize=0.01)
    # opt = ExtendedAdamOptimizer(stepsize=0.01)

    # print(train_test(opt, steps=120))

    # example values for layers = 15
    # steps | dropout = False | dropout = True
    # 20:     0.99600,          0.91500
    # 50:     0.98900,          0.81700
    # steps are normalised with regard to # branches

    train_params = {
        "wires": 5,
        "layers": 5,
        "starting_layers": 5,
        "steps": 20,
        "dataset": "simple",
        "testing": True,
        "optimizer": "gd",
        "step_size": 0.01,
        "dropout": "eileen",
        "sim_local": True,
        "logging": True, 
        "percentage": 0.05,
        "epsilon": 0.01,
        "seed": 1337,
        "cost_span": 5
    }
    check_params(train_params)
    train(train_params)