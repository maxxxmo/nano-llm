from src.model.neuron import Module
from src.model.layer import Layer
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
    