from maskit.datasets.utils import DataSet
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field

import random
import pennylane as qml
from pennylane import numpy as np

from maskit._masks import (
    DropoutMask,
    Mask,
    PerturbationAxis as Axis,
    PerturbationMode as Mode,
)
from maskit._masked_circuits import MaskedCircuit
from maskit.datasets import load_data
from maskit.utils import check_params
from maskit.circuits import cost, cost_basis, variational_circuit, basis_circuit
from maskit.log_results import log_results
from maskit.optimizers import ExtendedOptimizers
from maskit.ensembles import (
    AdaptiveEnsemble,
    Ensemble,
    EnsembleResult,
)


def get_device(sim_local: bool, wires: int, shots: Optional[int] = None):
    assert sim_local, "Currently only local simulation is supported"
    if sim_local:
        dev = qml.device("default.qubit", wires=wires, shots=shots)
    return dev


def init_parameters(
    layers: int,
    current_layers: int,
    wires: int,
    default_value: Optional[float],
    dynamic_parameters: bool = True,
    mask_type: Type[Mask] = DropoutMask,
) -> MaskedCircuit:
    params_uniform = np.random.uniform(
        low=-np.pi, high=np.pi, size=(current_layers, wires)
    )
    params_zero = np.zeros((layers - current_layers, wires))
    params_combined = np.concatenate((params_uniform, params_zero))
    mc = MaskedCircuit.full_circuit(
        parameters=params_combined,
        layers=layers,
        wires=wires,
        default_value=default_value,
        entangling_mask=DropoutMask(shape=(layers, wires - 1)),
        dynamic_parameters=dynamic_parameters,
        mask_type=mask_type,
    )
    mc.mask(Axis.LAYERS, mask_type=mask_type)[current_layers:] = True
    return mc


@dataclass
class LoggingData:
    interval: int = 1
    costs: Dict = field(default_factory=dict)
    branch_selection: Dict = field(default_factory=dict)
    branch_cost: Dict = field(default_factory=lambda: {"netto": {}, "brutto": {}})
    branch_cost_step: Dict = field(default_factory=lambda: {"netto": {}, "brutto": {}})
    active_count: Dict = field(default_factory=dict)
    cost_values: List = field(default_factory=list)
    active_count_values: List = field(default_factory=list)

    def log_result(self, result: EnsembleResult, step):
        self.cost_values.append(result.cost)
        self.active_count_values.append(result.active)
        if result.ensemble:
            self.branch_selection[step] = result.branch_name
            self.branch_cost["brutto"][step] = result.brutto
            self.branch_cost["netto"][step] = result.netto
            self.branch_cost_step["brutto"][step] = result.brutto_steps
            self.branch_cost_step["netto"][step] = result.netto_steps
        if step % self.interval == 0:  # perform logging
            self._persist_average_cost(step)
            self._persist_average_dropout(step)

    def _persist_average_cost(self, step):
        self.costs[step] = np.average(self.cost_values)
        self.cost_values.clear()

    def _persist_average_dropout(self, step):
        self.active_count[step] = np.average(self.active_count_values)
        self.active_count_values.clear()


def train(
    wires: int = 1,
    wires_to_measure: Tuple[int, ...] = (0,),
    interpret: Tuple[int, ...] = (0,),
    layers: int = 1,
    starting_layers: Optional[int] = None,
    steps: int = 1000,
    sim_local: bool = True,
    shots: Optional[int] = None,
    step_size: Optional[float] = None,
    optimizer: ExtendedOptimizers = ExtendedOptimizers.GD,
    default_value: Optional[float] = None,
    log_interval: int = 5,
    ensemble_type: Type[Ensemble] = Ensemble,
    ensemble_kwargs: Optional[Dict] = None,
    data: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    mask_type: Type[Mask] = DropoutMask,
):
    log_data = LoggingData(interval=log_interval)

    # set up circuit, training, dataset
    dev = get_device(sim_local=sim_local, wires=wires, shots=shots)
    opt = optimizer.value(step_size) if step_size else optimizer.value()
    dropout_ensemble = ensemble_type(
        **(ensemble_kwargs if ensemble_kwargs is not None else {"dropout": None})
    )

    rotation_choices = [0, 1, 2]
    rotations = [np.random.choice(rotation_choices) for _ in range(layers * wires)]

    current_layers = starting_layers if starting_layers else layers

    if data is not None and target is not None:
        circuit = qml.QNode(basis_circuit, dev)

        def cost_fn(params, masked_circuit=None):
            return cost_basis(
                circuit,
                params,
                current_data,
                current_target,
                rotations,
                masked_circuit,
                wires,
                wires_to_measure,
                interpret,
            )

    else:  # learning without encoding any data
        circuit = qml.QNode(variational_circuit, dev)

        def cost_fn(params, masked_circuit=None):
            return cost(
                circuit,
                params,
                rotations,
                masked_circuit,
            )

    # set up parameters
    masked_circuit = init_parameters(
        layers,
        current_layers,
        wires,
        default_value,
        dynamic_parameters=False if optimizer == ExtendedOptimizers.ADAM else True,
        mask_type=mask_type,
    )

    # -----------------------------
    # ======= TRAINING LOOP =======
    # -----------------------------
    for step in range(steps):
        if data is not None and target is not None:
            current_data = data[step % len(data)]
            current_target = target[step % len(target)]

        # TODO: add logging for adaptive ensembles
        result = dropout_ensemble.step(masked_circuit, opt, cost_fn, ensemble_steps=1)
        masked_circuit = result.branch
        log_data.log_result(result, step)

        if __debug__:
            print(
                f"Step: {step:4d} | Cost: {result.cost:.5f} |",
                # f"Gradient Variance: {np.var(gradient[0:current_layers]):.9f}",
            )

    if __debug__:
        print(masked_circuit.parameters)
        print(masked_circuit.full_mask(mask_type))

    return {
        "costs": log_data.costs,
        "final_cost": result.cost,
        "maximum_active": masked_circuit.parameters.size,
        "active": log_data.active_count,
        "branch_selections": log_data.branch_selection,
        "branch_costs": log_data.branch_cost,
        "branch_step_costs": log_data.branch_cost_step,
        "final_layers": current_layers,
        "params": masked_circuit.parameters.unwrap(),
        "mask": masked_circuit.full_mask(mask_type).unwrap(),
        "__rotations": rotations,
        "__masked_circuit": masked_circuit,
    }


def test(
    masked_circuit: MaskedCircuit,
    rotations: List,
    wires_to_measure: Tuple[int, ...],
    interpret: Tuple[int, ...],
    shots: Optional[int],
    sim_local: bool,
    data: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
):
    if data is None or target is None:
        pass
    elif data is not None and target is not None:
        dev = get_device(sim_local, wires=masked_circuit.wires, shots=shots)
        circuit = qml.QNode(basis_circuit, dev)
        correct = 0
        N = len(data)
        costs = []
        for current_data, current_target in zip(data, target):
            output = circuit(
                masked_circuit.differentiable_parameters,
                current_data,
                rotations,
                masked_circuit,
                masked_circuit.wires,
                wires_to_measure,
            )
            costs.append(
                cost_basis(
                    circuit,
                    masked_circuit.differentiable_parameters,
                    current_data,
                    current_target,
                    rotations,
                    masked_circuit,
                    masked_circuit.wires,
                    wires_to_measure,
                    interpret,
                )
            )
            selected_output = [
                elem.unwrap() for index, elem in enumerate(output) if index in interpret
            ]
            same = np.argmax(current_target) == np.argmax(selected_output)
            if same:
                correct += 1
            if __debug__:
                print(
                    f"Label: {current_target} Output: {selected_output} "
                    f"({output}) Correct: {same}"
                )

        accuracy = correct / N
        loss = np.average(costs)
        if __debug__:
            print(
                f"Accuracy = {correct} / {N} = {accuracy} \n",
                f"Avg Cost: {loss}",
            )
        return {
            "accuracy": accuracy,
            "loss": loss,
        }


if __name__ == "__main__":
    train_params = {
        "wires": 4,
        "wires_to_measure": [0, 1],
        "layers": 5,
        # "starting_layers": 10,  # only relevant if "dropout" == "growing"
        "steps": 1000,
        "dataset": "simple",
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
                            "mode": Mode.SET,
                            "axis": Axis.PARAMETERS,
                            "mask": DropoutMask,
                        },
                    },
                ],
                "right": [
                    {"copy": {}},
                    {
                        "perturb": {
                            "amount": 0.05,
                            "mode": Mode.RESET,
                            "axis": Axis.PARAMETERS,
                            "mask": DropoutMask,
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
        "mask_type": DropoutMask,
    }
    check_params(train_params)
    logging_activated = train_params.pop("logging", True)
    if logging_activated:
        train = log_results(
            train,
            exclude=(
                "data",
                "target",
            ),
        )
        test = log_results(
            test,
            exclude=(
                "data",
                "target",
            ),
        )
    seed = train_params.pop("seed", 1337)
    np.random.seed(seed)
    random.seed(seed)

    data_params = {
        "wires": train_params["wires"],
        "classes": [6, 9],
        "train_size": 120,
        "test_size": 100,
        "shuffle": True,
        "target_length": len(train_params.get("interpret", (0,))),
    }
    try:
        data = load_data(train_params.pop("dataset"), **data_params)
    except ValueError:
        data = DataSet(None, None, None, None)
    testing = train_params.pop("testing", False)
    result = train(**train_params, data=data.train_data, target=data.train_target)
    if testing:
        test(
            masked_circuit=result["__masked_circuit"],
            rotations=result["__rotations"],
            wires_to_measure=train_params.get("wires_to_measure", (0,)),
            interpret=train_params.get("interpret", (0,)),
            shots=train_params.get("shots", None),
            sim_local=train_params.get("sim_local", True),
            data=data.test_data,
            target=data.test_target,
        )
