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

class Layer(Module):
    """
    Args:
        Module (_type_): _description_
    """

    def __init__(self, nin, nout, dtype='float32', **kwargs):
        """
        Args:
            nin (_type_): _description_
            nout (_type_): _description_
        """
        self.neurons = [Neuron(nin, dtype=dtype, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Returns:
            _type_: _description_
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        Returns:
            _type_: _description_
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    Args:
        Module (_type_): _description_
    """

    def __init__(self, nin, nouts, dtype='float32'):
        """
        Args:
            nin (_type_): _description_
            nouts (_type_): _description_
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], dtype=dtype, nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns:
            _type_: _description_
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        Returns:
            _type_: _description_
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    