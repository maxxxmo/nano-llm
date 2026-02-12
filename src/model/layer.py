from src.model.neuron import Module
from src.model.neuron import Neuron
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