from collections import deque
from typing import Dict, NamedTuple, Optional
from pennylane import numpy as np

from maskit.masks import MaskedCircuit


class EnsembleResult(NamedTuple):
    #: branch that performs best
    branch: MaskedCircuit
    #: name of branch as configured
    branch_name: str
    #: cost of selected branch
    cost: float
    #: gradient of selected branch
    gradient: np.ndarray
    #: training steps including all other branches
    brutto_steps: int
    #: training steps for selected branch
    netto_steps: int
    #: training wrt count of parameters including all other branches
    brutto: int
    #: training wrt count of paramemters for selected branch
    netto: int
    #: True in case the ensemble was evaluated, otherwise False
    ensemble: bool


class Ensemble(object):
    __slots__ = ("dropout", "perturb")

    def __init__(self, dropout: Optional[Dict]):
        super().__init__()
        self.dropout = dropout
        self.perturb = True

    def _branch(
        self, masked_circuit: MaskedCircuit
    ) -> Optional[Dict[str, MaskedCircuit]]:
        if not self.perturb or self.dropout is None:
            return None
        branches = {}
        for key in self.dropout:
            branches[key] = MaskedCircuit.execute(masked_circuit, self.dropout[key])
        return branches

    def step(
        self,
        masked_circuit: MaskedCircuit,
        optimizer,
        objective_fn,
        *args,
        ensemble_steps: int = 0,
    ) -> EnsembleResult:
        """
        The parameter `ensemble_steps` defines the number of training steps that are
        executed for each ensemble branch in addition to one training step
        that is done before the branching.
        """
        # first one trainingstep
        params, _cost, _gradient = optimizer.step_cost_and_grad(
            objective_fn,
            *args,
            masked_circuit.parameters,
            masked_circuit=masked_circuit,
        )
        masked_circuit.parameters = params

        # then branching
        branches = self._branch(masked_circuit=masked_circuit)
        basic_active_gates = masked_circuit.active().unwrap()
        if branches is None:
            return EnsembleResult(
                branch=masked_circuit,
                branch_name="center",
                cost=objective_fn(
                    masked_circuit.parameters, masked_circuit=masked_circuit
                ).unwrap(),
                gradient=_gradient,
                brutto_steps=1,
                netto_steps=1,
                brutto=basic_active_gates,
                netto=basic_active_gates,
                ensemble=False,
            )
        branch_costs = []
        branch_gradients = []
        for branch in branches.values():
            for _ in range(ensemble_steps):
                params, _cost, _gradient = optimizer.step_cost_and_grad(
                    objective_fn, *args, branch.parameters, masked_circuit=branch
                )
                branch.parameters = params
            branch_costs.append(objective_fn(branch.parameters, masked_circuit=branch))
            branch_gradients.append(_gradient)
        minimum_index = branch_costs.index(min(branch_costs))
        branch_name = list(branches.keys())[minimum_index]
        selected_branch = branches[branch_name]
        # FIXME: as soon as real (in terms of masked) parameters are used,
        #   no mask has to be applied
        #   until then real gradients must be calculated as gradients also
        #   contain values from dropped gates
        return EnsembleResult(
            branch=selected_branch,
            branch_name=branch_name,
            cost=branch_costs[minimum_index].unwrap(),
            gradient=branch_gradients[minimum_index],
            brutto_steps=1 + len(branches) * ensemble_steps,
            netto_steps=1 + ensemble_steps,
            brutto=(
                basic_active_gates
                + sum(
                    [branch.active() * ensemble_steps for branch in branches.values()]
                )
            ).unwrap(),
            netto=(
                basic_active_gates + selected_branch.active() * ensemble_steps
            ).unwrap(),
            ensemble=True,
        )


class IntervalEnsemble(Ensemble):
    __slots__ = ("_interval", "_counter")

    def __init__(self, dropout: Optional[Dict], interval: int):
        super().__init__(dropout)
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval!r}")
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
        self,
        masked_circuit: MaskedCircuit,
        optimizer,
        objective_fn,
        *args,
        ensemble_steps: int = 1,
    ) -> EnsembleResult:
        self._counter += 1
        self._check_interval()
        return super().step(
            masked_circuit,
            optimizer,
            objective_fn,
            *args,
            ensemble_steps=ensemble_steps,
        )


class AdaptiveEnsemble(Ensemble):
    __slots__ = ("_cost", "epsilon")

    def __init__(
        self,
        dropout: Optional[Dict[str, Dict]],
        size: int,
        epsilon: float,
    ):
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size!r}")
        super().__init__(dropout)
        self._cost = deque(maxlen=size)
        self.epsilon = epsilon
        self.perturb = False

    def _check_cost(self, current_cost):
        self.perturb = False
        self._cost.append(current_cost)
        if self._cost.maxlen and len(self._cost) >= self._cost.maxlen:
            if current_cost > 0.1:  # evaluate current cost
                if np.sum(np.diff(self._cost)) < self.epsilon:
                    if __debug__:
                        print("======== allowing to perturb =========")
                    self._cost.clear()
                    self.perturb = True

    def step(
        self,
        masked_circuit: MaskedCircuit,
        optimizer,
        objective_fn,
        *args,
        ensemble_steps: int = 1,
    ) -> EnsembleResult:
        result = super().step(
            masked_circuit,
            optimizer,
            objective_fn,
            *args,
            ensemble_steps=ensemble_steps,
        )
        self._check_cost(result.cost)
        return result
