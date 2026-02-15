import numpy as np
from src.tensor_model.tensor import Tensor
class Module ():
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01)
        self.b = Tensor(np.zeros((1, out_features)))
    def __call__ (self,x):
        return x @ self.W + self.b
    def parameters(self):
        return [self.W, self.b]

class Softmax(Module):
    def __init(self, axis=-1):
        super().__init__()
        self.axis = axis
    def __call__(self, x):
            return x.softmax(axis=self.axis)
    def parameters(self):
        return []

class ReLU(Module):
    def __call__(self, x):
        return x.relu()

class GeLU(Module):
    def __call__(self, x):
        return x.gelu()

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, dim)))
        self.beta = Tensor(np.zeros((1, dim)))

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean)**2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / ((var + self.eps)**0.5)
        return self.gamma * x_norm + self.beta

    def parameters(self):
        return [self.gamma, self.beta]