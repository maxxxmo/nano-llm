from src.model.neuron import Module 

class Activation(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """
    def __init__(self):
        super().__init__()

    def parameters(self):
        return []

class ReLU(Activation):
    """_summary_

    Args:
        Activation (_type_): _description_
    """
    def __call__(self, x):
        if isinstance(x, list):
            return [v.relu() for v in x]
        return x.relu()

class Softmax(Activation):
    """_summary_

    Args:
        Activation (_type_): _description_
    """
    def __call__(self, x):
        assert isinstance(x, list), "Softmax s'applique sur une liste de Values"
        max_data = max(v.data for v in x)
        exps = [(v - max_data).exp() for v in x]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]
