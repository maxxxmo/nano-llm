import numpy as np
from src.tensor_model.nn import Linear
from src.tensor_model.nn import Module
from src.tensor_model.nn import LayerNorm

class MLP(Module): 
    """_summary_

    Args:
        Module (_type_): _description_
    """
    def __init__(self, n_embd):
        """_summary_

        Args:
            n_embd (_type_): _description_
        """
        self.ln      = LayerNorm(n_embd)
        self.c_fc    = Linear(n_embd, 4 * n_embd)
        self.c_proj  = Linear(4 * n_embd, n_embd) 
        
    def __call__(self, x):
        """
        Docstring for __call__
        
        :param self: Description
        :param x: Description
        """  
        res= self.ln(x)
        res = self.c_fc(res)
        res = res.relu() 
        res = self.c_proj(res)
        return x + res

    def parameters(self):
        return self.c_fc.parameters() + self.c_proj.parameters() + self.ln.parameters()
