from typing import List

import pennylane as qml
from pennylane import numpy as np
from collections import deque

from maskit.masked_parameters import (
    MaskedParameters,
    PerturbationMode,
)
from maskit.iris import load_iris
from maskit.utils import cross_entropy, check_params
from maskit.circuits import variational_circuit, iris_circuit
from maskit.log_results import log_results
from maskit.optimizers import ExtendedAdamOptimizer, ExtendedGradientDescentOptimizer


def get_device(sim_local, wires, analytic=True):
    assert sim_local, "Currently only local simulation is supported"
    if sim_local:
        dev = qml.device("default.qubit", wires=wires, analytic=analytic)
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
                *args, params, mask=branch.mask
            )
        branch.params = params
        branch_costs.append(args[0](params, mask=branch.mask))
        gradients.append(gradient)
    minimum_cost = min(branch_costs)
    minimum_index = branch_costs.index(minimum_cost)
    return (
        branches[minimum_index],
        branch_costs[minimum_index],
        gradients[minimum_index],
    )


@log_results
def train(train_params):
    logging_costs = {}
    logging_branches = {}
    logging_branch_selection = {}
    logging_branch_enforcement = {}
    logging_gate_count = {}
    logging_cost_values = []
    logging_gate_count_values = []

    np.random.seed(train_params["seed"])

    # set up circuit, training, dataset
    wires = train_params["wires"]
    layers = train_params["layers"]
    dev = get_device(train_params["sim_local"], wires=wires)

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    current_layers = (
        layers
        if train_params["dropout"] is not ["growing"]
        else train_params["starting_layers"]
    )

    if train_params["optimizer"] == "gd":
        opt = ExtendedGradientDescentOptimizer(train_params["step_size"])
    elif train_params["optimizer"] == "adam":
        opt = ExtendedAdamOptimizer(train_params["step_size"])

    steps = train_params["steps"]

    if train_params["dataset"] == "simple":
        circuit = qml.QNode(variational_circuit, dev)

        def cost_fn(params, mask=None):
            return cost(circuit, params, wires, current_layers, rotations, mask)

    elif train_params["dataset"] == "iris":
        circuit = qml.QNode(iris_circuit, dev)
        x_train, y_train, x_test, y_test = load_iris()

        def cost_fn(params, mask=None):
            return cost_iris(
                circuit, params, data, target, wires, current_layers, rotations, mask
            )

    # set up parameters
    params_uniform = np.random.uniform(
        low=-np.pi, high=np.pi, size=(current_layers, wires)
    )
    params_zero = np.zeros((layers - current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    masked_params = MaskedParameters(params_combined)

    if train_params["dropout"] == "eileen":
        # masked_params.perturb(int(layers * wires * 0.5))
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
            masked_params.perturb(
                masked_params.params.size // 10, mode=PerturbationMode.ADD
            )
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
                logging_branches[step] = {
                    "center": "No perturbation",
                    "left": {
                        "amount": 1,
                        "mode": PerturbationMode.ADD,
                        "axis": left_branch.perturbation_axis,
                    },
                    "right": {
                        "amount": amount,
                        "mode": PerturbationMode.REMOVE,
                        "axis": right_branch.perturbation_axis,
                    },
                }
            else:
                branches = [masked_params]
        else:
            branches = [masked_params]

        if train_params["dataset"] == "iris":
            data = x_train[step % len(x_train)]
            target = y_train[step % len(y_train)]

        masked_params, current_cost, gradient = ensemble_step(branches, opt, cost_fn)
        branch_index = branches.index(masked_params)
        logging_branch_selection[step] = (
            "center" if branch_index == 0 else "left" if branch_index == 1 else "right"
        )

        logging_cost_values.append(current_cost.unwrap())
        logging_gate_count_values.append(np.sum(masked_params.mask))
        if step % train_params["log_interval"] == 0:
            # perform logging
            logging_costs[step] = np.average(logging_cost_values)
            logging_gate_count[step] = np.average(logging_gate_count_values)
            logging_cost_values.clear()
            logging_gate_count_values.clear()

        # get the real gradients as gradients also contain values from dropped gates
        real_gradients = masked_params.apply_mask(gradient)

        print(
            "Step: {:4d} | Cost: {: .5f} | Gradient Variance: {: .9f}".format(
                step, current_cost, np.var(real_gradients[0:current_layers])
            )
        )

        if train_params["dropout"] == "eileen":
            costs.append(current_cost)
            if len(costs) >= train_params["cost_span"] and current_cost > 0.1:
                if (
                    sum(
                        [
                            abs(cost - costs[index + 1])
                            for index, cost in enumerate(list(costs)[:-1])
                        ]
                    )
                    < train_params["epsilon"]
                ):
                    print("======== allowing to perturb =========")
                    if np.sum(masked_params.mask) >= layers * wires * 0.3:
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                        logging_branch_enforcement[step + 1] = {
                            "amount": 1,
                            "mode": PerturbationMode.REMOVE,
                            "axis": masked_params.perturbation_axis,
                        }
                    elif (
                        current_cost < 0.25
                        and np.sum(masked_params.mask) >= layers * wires * 0.05
                    ):
                        masked_params.perturb(1, mode=PerturbationMode.REMOVE)
                        logging_branch_enforcement[step + 1] = {
                            "amount": 1,
                            "mode": PerturbationMode.REMOVE,
                            "axis": masked_params.perturbation_axis,
                        }
                    costs.clear()
                    perturb = True

    print(masked_params.params)
    print(masked_params.mask)

    return {
        "costs": logging_costs,
        "final_cost": current_cost.unwrap(),
        "branch_enforcements": logging_branch_enforcement,
        "dropouts": logging_gate_count,
        "branches": logging_branches,
        "branch_selections": logging_branch_selection,
        "final_layers": current_layers,
        "params": masked_params.params.unwrap(),
        "mask": masked_params.mask.unwrap(),
        "__rotations": rotations,
    }


def test(train_params, params, mask, layers, rotations):
    if train_params["dataset"] == "simple":
        pass
    elif train_params["dataset"] == "iris":
        wires = train_params["wires"]
        dev = get_device(train_params["sim_local"], wires=wires)
        circuit = qml.QNode(iris_circuit, dev)
        _, _, x_test, y_test = load_iris()
        correct = 0
        N = len(x_test)
        costs = []
        for _step, (data, target) in enumerate(zip(x_test, y_test)):
            test_mask = np.zeros_like(params, dtype=bool, requires_grad=False)
            output = circuit(
                params,
                data,
                wires,
                layers,
                rotations,
                test_mask,
            )
            c = cost_iris(
                circuit,
                params,
                data,
                target,
                wires,
                layers,
                rotations,
                test_mask,
            )
            costs.append(c)
            same = np.argmax(target) == np.argmax(output)
            if same:
                correct += 1
            print("Label: {} Output: {} Correct: {}".format(target, output, same))
        print(
            "Accuracy = {} / {} = {} \nAvg Cost: {}".format(
                correct, N, correct / N, np.average(costs)
            )
        )


if __name__ == "__main__":
    train_params = {
        "wires": 10,
        "layers": 5,
        "starting_layers": 10,  # only relevant if "dropout" == "growing"
        "steps": 1000,
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
        "cost_span": 5,
        "log_interval": 5,
    }
    check_params(train_params)
    result = train(train_params)
    if train_params["testing"]:
        test(
            train_params,
            result["params"],
            result["mask"],
            result["final_layers"],
            result["__rotations"],
        )
