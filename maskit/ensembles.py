from collections import deque
from typing import Dict, List, Optional
import pennylane.numpy as np

from maskit.masks import PerturbationAxis, PerturbationMode, MaskedCircuit

ENFORCEMENT = [
    {
        "perturb": {
            "axis": PerturbationAxis.RANDOM,
            "amount": 1,
            "mode": PerturbationMode.REMOVE,
        }
    }
]


class Ensemble(object):
    __slots__ = ("dropout", "perturb")

    def __init__(self, dropout: Optional[Dict], *args, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.perturb = True

    def _branch(self, masked_circuit: MaskedCircuit) -> Dict[str, MaskedCircuit]:
        if not self.perturb or self.dropout is None:
            return {"center": masked_circuit}
        branches = {}
        for key in self.dropout:
            branches[key] = MaskedCircuit.execute(masked_circuit, self.dropout[key])
        return branches

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 1
    ):
        # TODO: step_count is undefined, does it mean how often it is disturbed or
        #   how often the disturbance is trained before being evaluated
        # first one trainingstep
        params, _cost, gradient = optimizer.step_cost_and_grad(
            *args, masked_circuit.parameters, masked_circuit=masked_circuit
        )
        masked_circuit.parameters = params

        # then branching
        branches = self._branch(masked_circuit=masked_circuit)
        branch_costs = []
        for branch in branches.values():
            branch_costs.append(args[0](branch.parameters, masked_circuit=branch))

        minimum_index = branch_costs.index(min(branch_costs))
        branch_name = list(branches.keys())[minimum_index]
        selected_branch = branches[branch_name]
        # FIXME: as soon as real (in terms of masked) parameters are used,
        #   no mask has to be applied
        #   until then real gradients must be calculated as gradients also
        #   contain values from dropped gates
        return (selected_branch, branch_name, branch_costs[minimum_index], 0)


class IntervalEnsemble(Ensemble):
    __slots__ = ("_interval", "_counter")

    def __init__(self, dropout: Optional[Dict], interval: int):
        super().__init__(dropout)
        self._interval = interval
        self._counter = 0
        self.perturb = False

    def _check_interval(self):
        if self._counter % self._interval == 0:
            self.perturb = True
            self._counter = 0  # reset the counter
        else:
            self.perturb = False

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 1
    ):
        self._counter += 1
        self._check_interval()
        result = super().step(masked_circuit, optimizer, *args, step_count=step_count)
        return result


class AdaptiveEnsemble(Ensemble):
    __slots__ = ("_cost", "epsilon", "enforcement_dropout", "perturb")

    def __init__(
        self,
        dropout: Optional[Dict[str, Dict]],
        size: int,
        epsilon: float,
        enforcement_dropout: List[Dict],
    ):
        if size <= 0:
            raise ValueError(f"Size must be bigger than 0 (received {size})")
        super().__init__(dropout)
        self._cost = deque(maxlen=size)
        self.epsilon = epsilon
        self.enforcement_dropout = enforcement_dropout
        self.perturb = False

    def _branch(self, masked_circuit: MaskedCircuit) -> Dict[str, MaskedCircuit]:
        result = super()._branch(masked_circuit)
        self.perturb = False
        return result

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 1
    ):
        branch, branch_name, branch_cost, gradients = super().step(
            masked_circuit, optimizer, *args, step_count=step_count
        )
        self._cost.append(branch_cost)
        if self._cost.maxlen and len(self._cost) >= self._cost.maxlen:
            if branch_cost > 0.1:  # evaluate current cost
                if (
                    sum(
                        [
                            abs(cost - self._cost[index + 1])
                            for index, cost in enumerate(list(self._cost)[:-1])
                        ]
                    )
                    < self.epsilon
                ):
                    if __debug__:
                        print("======== allowing to perturb =========")
                    if (
                        np.sum(branch.mask)
                        >= branch.layer_mask.size * branch.wire_mask.size * 0.3
                    ):
                        branch = MaskedCircuit.execute(branch, self.enforcement_dropout)
                        # logging_branch_enforcement[step + 1] = {  # TODO
                        #     "amount": 1,
                        #     "mode": PerturbationMode.REMOVE,
                        #     "axis": PerturbationAxis.RANDOM,
                        # }
                    elif (
                        branch_cost < 0.25
                        and np.sum(branch.mask)
                        >= branch.layer_mask.size * branch.wire_mask.size * 0.05
                    ):
                        branch = MaskedCircuit.execute(branch, self.enforcement_dropout)
                        # logging_branch_enforcement[step + 1] = {  # TODO
                        #     "amount": 1,
                        #     "mode": PerturbationMode.REMOVE,
                        #     "axis": PerturbationAxis.RANDOM,
                        # }
                    self._cost.clear()
                    self.perturb = True
        return (branch, branch_name, branch_cost, gradients)
