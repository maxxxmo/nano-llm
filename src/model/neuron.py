import numpy as np

def Neuron (): 
    def __init__ (self, n_inputs):
        self.weight = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        
    def predict(self, inputs):
        output = self.weights @ inputs + self.bias
        return output