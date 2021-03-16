from typing import List, Optional

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
from maskit.optimizers import ExtendedOptimizers


def get_device(sim_local: bool, wires: int, analytic: bool = True):
    assert sim_local, "Currently only local simulation is supported"
    if sim_local:
        dev = qml.device("default.qubit", wires=wires, analytic=analytic)
    return dev


def cost(circuit, params, wires: int, layers: int, rotations: List, dropouts):
    return 1 - circuit(params, wires, layers, rotations, dropouts)[0]


def cost_iris(
    circuit, params, data, target, wires: int, layers: int, rotations: List, dropouts
):
    prediction = circuit(params, data, wires, layers, rotations, dropouts)
    return cross_entropy(predictions=prediction, targets=target)


def ensemble_step(branches: List[MaskedParameters], optimizer, *args, step_count=1):
    branch_costs = []
    gradients = []
    for branch in branches:
        params = branch.params
        for _ in range(step_count):
            params, _cost, gradient = optimizer.step_cost_and_grad(
                *args, params, mask=branch.mask
            )
        branch.params = params
        branch_costs.append(args[0](params, mask=branch.mask))
        gradients.append(gradient)
    minimum_index = branch_costs.index(min(branch_costs))
    return (
        branches[minimum_index],
        branch_costs[minimum_index],
        gradients[minimum_index],
    )


def ensemble_branches(dropout, masked_params, amount: int = 1, perturb: bool = True):
    if dropout == "random":
        left_branch = masked_params.copy()
        right_branch = masked_params.copy()
        # randomly perturb branches
        left_branch.perturb(1, mode=PerturbationMode.REMOVE)
        right_branch.perturb()
        branches = [masked_params, left_branch, right_branch]
        description = {
            "center": "No perturbation",
            "left": {
                "amount": 1,
                "mode": PerturbationMode.REMOVE,
                "axis": left_branch.perturbation_axis,
            },
            "right": {
                "amount": None,
                "mode": PerturbationMode.INVERT,
                "axis": right_branch.perturbation_axis,
            },
        }
    elif dropout == "classical":
        masked_params.reset()
        masked_params.perturb(
            masked_params.params.size // 10, mode=PerturbationMode.ADD
        )
        branches = [masked_params]
        description = {
            "center": {
                "amount": masked_params.params.size // 10,
                "mode": PerturbationMode.ADD,
                "axis": masked_params.perturbation_axis,
                # TODO: added on empty mask...
            },
        }
    elif dropout == "eileen" and perturb:
        left_branch = masked_params.copy()
        right_branch = masked_params.copy()
        left_branch.perturb(amount=1, mode=PerturbationMode.ADD)
        right_branch.perturb(amount=amount, mode=PerturbationMode.REMOVE)
        branches = [masked_params, left_branch, right_branch]
        description = {
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
        description = {
            "center": "No perturbation",
        }
    return branches, description


def init_parameters(layers: int, current_layers: int, wires: int):
    params_uniform = np.random.uniform(
        low=-np.pi, high=np.pi, size=(current_layers, wires)
    )
    params_zero = np.zeros((layers - current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    return MaskedParameters(params_combined)


def train(
    train_params, train_data: Optional[List] = None, train_target: Optional[List] = None
):
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
    steps = train_params["steps"]
    dev = get_device(train_params["sim_local"], wires=wires)
    opt = train_params["optimizer"].value(train_params["step_size"])

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    current_layers = (
        layers
        if train_params["dropout"] != "growing"
        else train_params["starting_layers"]
    )

    if train_params["dataset"] == "simple":
        circuit = qml.QNode(variational_circuit, dev)

        def cost_fn(params, mask=None):
            return cost(circuit, params, wires, current_layers, rotations, mask)

    elif train_params["dataset"] == "iris":
        circuit = qml.QNode(iris_circuit, dev)

        def cost_fn(params, mask=None):
            return cost_iris(
                circuit, params, data, target, wires, current_layers, rotations, mask
            )

    # set up parameters
    masked_params = init_parameters(layers, current_layers, wires)

    if train_params["dropout"] == "eileen":
        # masked_params.perturb(int(layers * wires * 0.5))
        amount = int(wires * layers * train_params["percentage"])
        perturb = False
        costs = deque(maxlen=5)

    # -----------------------------
    # ======= TRAINING LOOP =======
    # -----------------------------
    for step in range(steps):
        if train_params["dropout"] == "growing":
            # TODO useful condition
            # maybe combine with other dropouts
            if step > 0 and step % 1000 == 0:
                current_layers += 1
        branches, description = ensemble_branches(
            train_params["dropout"], masked_params, amount, perturb=perturb
        )
        perturb = False
        logging_branches[step] = description

        if train_params["dataset"] == "iris":
            data = train_data[step % len(train_data)]
            target = train_target[step % len(train_target)]

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

        if __debug__:
            print(
                f"Step: {step:4d} | Cost: {current_cost:.5f} |",
                f"Gradient Variance: {np.var(real_gradients[0:current_layers]):.9f}",
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
                    if __debug__:
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

    if __debug__:
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


def test(
    train_params,
    params,
    mask,
    layers: int,
    rotations: List,
    test_data: Optional[List] = None,
    test_target: Optional[List] = None,
):
    if train_params["dataset"] == "simple":
        pass
    elif train_params["dataset"] == "iris":
        wires = train_params["wires"]
        dev = get_device(train_params["sim_local"], wires=wires)
        circuit = qml.QNode(iris_circuit, dev)
        correct = 0
        N = len(test_data)
        costs = []
        for _step, (data, target) in enumerate(zip(test_data, test_target)):
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
            if __debug__:
                print(f"Label: {target} Output: {output} Correct: {same}")
        if __debug__:
            print(
                f"Accuracy = {correct} / {N} = {correct/N} \n",
                f"Avg Cost: {np.average(costs)}",
            )


if __name__ == "__main__":
    train_params = {
        "wires": 10,
        "layers": 5,
        "starting_layers": 10,  # only relevant if "dropout" == "growing"
        "steps": 1000,
        "dataset": "simple",
        "testing": True,
        "optimizer": ExtendedOptimizers.GD,
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
    if train_params.get("logging", True):
        train = log_results(train)
    train_data, train_target, test_data, test_target = (
        load_iris() if train_params["dataset"] == "iris" else [None, None, None, None]
    )
    result = train(train_params, train_data=train_data, train_target=train_target)
    if train_params["testing"]:
        test(
            train_params,
            result["params"],
            result["mask"],
            result["final_layers"],
            result["__rotations"],
            test_data=test_data,
            test_target=test_target,
        )
