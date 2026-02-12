import random
from src.model.core import Value


class Module:
    """
    """
    def zero_grad(self):
        """
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Returns:
            _type_: _description_
        """
        return []

class Neuron(Module):
    """
    Args:
        Module (_type_): _description_
    """

    def __init__(self, nin, nonlin=True, dtype='float32'):
        """
        Args:
            nin (_type_): _description_
            nonlin (bool, optional): _description_. Defaults to True.
        """
        self.w = [Value(random.uniform(-1,1), dtype=dtype) for _ in range(nin)]
        self.b = Value(0, dtype=dtype)
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        """
        Returns:
            _type_: _description_
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Returns:
            _type_: _description_
        """
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
