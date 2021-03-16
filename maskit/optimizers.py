from pennylane import GradientDescentOptimizer, AdamOptimizer
from enum import Enum


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
