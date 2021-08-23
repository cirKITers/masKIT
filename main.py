from typing import List, Optional

import random
import pennylane as qml
from pennylane import numpy as np

from maskit.masks import MaskedCircuit, PerturbationAxis, PerturbationMode

from maskit.examples.load_data import load_data
from maskit.utils import cross_entropy, check_params
from maskit.circuits import variational_circuit, iris_circuit
from maskit.log_results import log_results
from maskit.optimizers import ExtendedOptimizers
from maskit.ensembles import (
    AdaptiveEnsemble,
    Ensemble,
)


def get_device(sim_local: bool, wires: int, shots: Optional[int] = None):
    assert sim_local, "Currently only local simulation is supported"
    if sim_local:
        dev = qml.device("default.qubit", wires=wires, shots=shots)
    return dev


def cost(
    circuit,
    params,
    rotations: List,
    masked_circuit: MaskedCircuit,
):
    return 1 - circuit(params, rotations, masked_circuit)[0]


def cost_iris(
    circuit,
    params,
    data,
    target,
    rotations: List,
    masked_circuit: MaskedCircuit,
):
    prediction = circuit(params, data, rotations, masked_circuit)
    return cross_entropy(predictions=prediction, targets=target)


def init_parameters(
    layers: int, current_layers: int, wires: int, default_value: Optional[float]
) -> MaskedCircuit:
    params_uniform = np.random.uniform(
        low=-np.pi, high=np.pi, size=(current_layers, wires)
    )
    params_zero = np.zeros((layers - current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    mc = MaskedCircuit(
        parameters=params_combined,
        layers=layers,
        wires=wires,
        default_value=default_value,
    )
    mc.layer_mask[current_layers:] = True
    return mc


def train(
    train_params, train_data: Optional[List] = None, train_target: Optional[List] = None
):
    logging_costs = {}
    logging_branch_selection = {}
    logging_branch_cost = {"netto": {}, "brutto": {}}
    logging_branch_cost_step = {"netto": {}, "brutto": {}}
    logging_dropout_count = {}
    logging_cost_values = []
    logging_dropout_count_values = []

    np.random.seed(train_params["seed"])
    random.seed(train_params["seed"])

    # set up circuit, training, dataset
    wires = train_params["wires"]
    layers = train_params["layers"]
    steps = train_params.get("steps", 1000)
    dev = get_device(train_params.get("sim_local", True), wires=wires)
    step_size = train_params.get("step_size", None)
    if step_size:
        opt = train_params["optimizer"].value(step_size)
    else:
        opt = train_params["optimizer"].value()
    dropout_ensemble = train_params.get("ensemble_type", Ensemble)(
        **train_params.get("ensemble_kwargs", {"dropout": None})
    )

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    current_layers = train_params.get("starting_layers", layers)
    default_value = train_params.get("default_value", None)

    if train_params["dataset"] == "simple":
        circuit = qml.QNode(variational_circuit, dev)

        def cost_fn(params, masked_circuit=None):
            return cost(
                circuit,
                params,
                rotations,
                masked_circuit,
            )

    elif (
        train_params["dataset"] == "iris"
        or train_params["dataset"] == "mnist"
        or train_params["dataset"] == "circles"
    ):
        # TODO: probably needs some refactoring for the other datasets
        circuit = qml.QNode(iris_circuit, dev)

        def cost_fn(params, masked_circuit=None):
            return cost_iris(
                circuit,
                params,
                data,
                target,
                rotations,
                masked_circuit,
            )

    # set up parameters
    masked_circuit = init_parameters(layers, current_layers, wires, default_value)

    # -----------------------------
    # ======= TRAINING LOOP =======
    # -----------------------------
    for step in range(steps):
        if (
            train_params["dataset"] == "iris"
            or train_params["dataset"] == "mnist"
            or train_params["dataset"] == "circles"
        ):
            data = train_data[step % len(train_data)]
            target = train_target[step % len(train_target)]

        # TODO: add logging for adaptive ensembles
        result = dropout_ensemble.step(masked_circuit, opt, cost_fn, ensemble_steps=1)
        masked_circuit = result.branch
        if result.ensemble:
            logging_branch_selection[step] = result.branch_name
            logging_branch_cost["brutto"][step] = result.brutto
            logging_branch_cost["netto"][step] = result.netto
            logging_branch_cost_step["brutto"][step] = result.brutto_steps
            logging_branch_cost_step["netto"][step] = result.netto_steps
        logging_cost_values.append(result.cost)
        logging_dropout_count_values.append(np.sum(masked_circuit.mask))
        if step % train_params["log_interval"] == 0:
            # perform logging
            logging_costs[step] = np.average(logging_cost_values)
            logging_dropout_count[step] = np.average(logging_dropout_count_values)
            logging_cost_values.clear()
            logging_dropout_count_values.clear()

        if __debug__:
            print(
                f"Step: {step:4d} | Cost: {result.cost:.5f} |",
                # f"Gradient Variance: {np.var(gradient[0:current_layers]):.9f}",
            )

    if __debug__:
        print(masked_circuit.parameters)
        print(masked_circuit.mask)

    return {
        "costs": logging_costs,
        "final_cost": result.cost,
        "dropouts": logging_dropout_count,
        "branch_selections": logging_branch_selection,
        "branch_costs": logging_branch_cost,
        "branch_step_costs": logging_branch_cost_step,
        "final_layers": current_layers,
        "params": masked_circuit.parameters.unwrap(),
        "mask": masked_circuit.mask.unwrap(),
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
    elif (
        train_params["dataset"] == "iris"
        or train_params["dataset"] == "mnist"
        or train_params["dataset"] == "circles"
    ):
        # TODO: probably needs some refactoring for the other datasets
        wires = train_params["wires"]
        dev = get_device(train_params["sim_local"], wires=wires)
        circuit = qml.QNode(iris_circuit, dev)
        correct = 0
        N = len(test_data)
        costs = []
        masked_circuit = MaskedCircuit(parameters=params, layers=layers, wires=wires)
        for _step, (data, target) in enumerate(zip(test_data, test_target)):
            output = circuit(
                params,
                data,
                rotations,
                masked_circuit,
            )
            c = cost_iris(
                circuit,
                params,
                data,
                target,
                rotations,
                masked_circuit,
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
        "wires": 4,
        "layers": 5,
        # "starting_layers": 10,  # only relevant if "dropout" == "growing"
        "steps": 100,
        "dataset": "iris",
        "testing": True,
        "ensemble_type": AdaptiveEnsemble,
        "ensemble_kwargs": {
            "dropout": {
                "center": None,
                "left": [
                    {"copy": {}},
                    {
                        "perturb": {
                            "amount": 1,
                            "mode": PerturbationMode.ADD,
                            "axis": PerturbationAxis.RANDOM,
                        },
                    },
                ],
                "right": [
                    {"copy": {}},
                    {
                        "perturb": {
                            "amount": 0.05,
                            "mode": PerturbationMode.REMOVE,
                            "axis": PerturbationAxis.RANDOM,
                        }
                    },
                ],
            },
            "size": 5,
            "epsilon": 0.01,
        },
        "optimizer": ExtendedOptimizers.GD,
        "step_size": 0.01,
        "sim_local": True,
        "logging": True,
        "seed": 1337,
        "log_interval": 5,
    }
    check_params(train_params)
    if train_params.get("logging", True):
        train = log_results(train)

    data_params = {
        "wires": train_params["wires"],
        "embedding": None,
        "classes": [6, 9],
        "train_size": 120,
        "test_size": 100,
        "shuffle": True,
    }
    train_data, train_target, test_data, test_target = load_data(
        train_params["dataset"], train_params["wires"], None, data_params
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
