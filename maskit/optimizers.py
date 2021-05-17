from typing import Tuple
from pennylane import GradientDescentOptimizer, AdamOptimizer
from pennylane._grad import grad as get_gradient
from enum import Enum
from pennylane.numpy import ndarray
import scipy.optimize as sciopt


class L_BFGS_B:
    __slots__ = (
        "bounds",
        "m",
        "factr",
        "pgtol",
        "epsilon",
        "iprint",
        "maxfun",
        "maxiter",
        "disp",
        "callback",
        "maxls",
    )

    def __init__(
        self,
        bounds=None,
        m: int = 10,
        factr: float = 1e7,  # qiskit: factr: float = 10,
        pgtol: float = 1e-5,
        epsilon: float = 1e-8,  # qiskit: epsilon: float = 1e-08,
        iprint: int = -1,  # qiskit: iprint: int = -1,
        maxfun: int = 15000,  # qiskit: maxfun: int = 1000,
        maxiter: int = 15000,  # qiskit: maxiter: int = 15000,
        disp=None,
        callback=None,
        maxls: int = 20,
    ):
        """
        In case the method :py:method:`~.step` is used, the value of parameter
        `maxiter` is ignored and interpreted as `1` instead.

        :param bounds: [description], defaults to None
        :param m: [description], defaults to 10
        :param factr: [description], defaults to 1e7
        :param pgtol: [description], defaults to 1e-5
        :param epsilon: [description], defaults to 1e-8
        :param iprint: [description], defaults to -1
        :param maxfun: [description], defaults to 15000
        :param maxiter: [description], defaults to 15000
        :param disp: [description], defaults to None
        :param callback: [description], defaults to None
        :param maxls: [description], defaults to 20
        """
        self.bounds = bounds
        self.m = m
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.iprint = iprint
        self.maxfun = maxfun
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.maxls = maxls

    def optimize(
        self,
        objective_fn,
        parameters: ndarray,
        *args,
        grad_fn=None,
        **kwargs,
    ):
        return self._optimise(
            objective_fn, parameters, self.maxiter, *args, grad_fn=grad_fn, **kwargs
        )

    def _optimise(
        self, objective_fn, parameters: ndarray, maxiter, *args, grad_fn, **kwargs
    ):
        shape = parameters.shape
        shaped_fn = self._wrap_objective_fn(objective_fn, shape, **kwargs)
        approx_grad = False  # TODO: check the defaults of pennylane optimisers
        sol, opt, info = sciopt.fmin_l_bfgs_b(
            shaped_fn,
            parameters.flatten(),
            bounds=self.bounds,
            fprime=get_gradient(shaped_fn) if grad_fn is None else grad_fn,
            approx_grad=approx_grad,
            maxiter=1,
        )
        return sol.reshape(shape), opt, info["grad"]

    def _wrap_objective_fn(self, objective_fn, shape, **kwargs):
        return lambda params, *args: objective_fn(
            params.reshape(shape), *args, **kwargs
        )

    def step(self, objective_fn, parameters, *args, grad_fn=None, **kwargs) -> ndarray:
        sol, _, _ = self._optimise(
            objective_fn, parameters, 1, *args, grad_fn=grad_fn, **kwargs
        )
        return sol

    def step_and_cost(
        self, objective_fn, parameters, *args, grad_fn=None, **kwargs
    ) -> Tuple[ndarray, float]:
        sol, cost, _ = self._optimise(
            objective_fn, parameters, 1, *args, grad_fn=grad_fn, **kwargs
        )
        return sol, cost

    def step_cost_and_grad(
        self, objective_fn, parameters, *args, grad_fn=None, **kwargs
    ):
        sol, cost, grad = self._optimise(
            objective_fn, parameters, 1, *args, grad_fn=grad_fn, **kwargs
        )
        return sol, cost, grad


class ExtendedGradientDescentOptimizer(GradientDescentOptimizer):
    def step_cost_and_grad(self, objective_fn, *args, grad_fn=None, **kwargs):
        """
        This function copies the functionality of the GradientDescentOptimizer
        one-to-one but changes the return statement to also return the gradient.
        """
        gradient, forward = self.compute_grad(
            objective_fn, args, kwargs, grad_fn=grad_fn
        )
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


class ExtendedOptimizers(Enum):
    GD = ExtendedGradientDescentOptimizer
    ADAM = ExtendedAdamOptimizer
    L_BFGS_B = L_BFGS_B
