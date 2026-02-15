import numpy as np
from src.tensor_model.nn import Linear
from src.tensor_model.nn import Module
    
class MLP(Module): 
    def __init__(self, n_embd):
        self.c_fc    = Linear(n_embd, 4 * n_embd)
        self.c_proj  = Linear(4 * n_embd, n_embd) 
        
    def __call__(self, x):  # explain layers and projections?
        x = self.c_fc(x)
        x = x.relu() 
        x = self.c_proj(x)
        return x

    def parameters(self):
        return self.c_fc.parameters() + self.c_proj.parameters()