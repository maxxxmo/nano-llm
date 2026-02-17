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
    
class SelfAttention(Module):
    def __init__(self, d_model, max_seq_len=1024):
        self.d_model = d_model
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.mask = Tensor(np.tril(np.ones((max_seq_len, max_seq_len))))
    def __call__(self, x):
        T =x.shape[1]
        q = self.w_q(x) 
        k = self.w_k(x)
        v = self.w_v(x)
        k_t = k.transpose(-1, -2)
        scores = (q @ k_t) / (self.d_model ** 0.5)
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        weights = scores.softmax(axis=-1)
        out = weights @ v
        return out

    def parameters(self):
        return self.w_q.parameters() + self.w_k.parameters() + self.w_v.parameters()
    
class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024):
        self.d_model = d_model
        self.n_heads = n_heads
        # La dimension de chaque tête doit être un entier
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.d_k = d_model // n_heads
        
        # On projette toujours en (d_model, d_model), on "splintera" après
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        
        # La couche de sortie (Wo) qui fusionne les têtes
        self.w_o = Linear(d_model, d_model)
        
        # Le masque causal (tril) reste identique
        self.mask = Tensor(np.tril(np.ones((max_seq_len, max_seq_len))))

    def __call__(self, x):
        """reminder
        B->batch size
        T->sequence len
        C->

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        B, T, C = x.shape 
        assert C == self.d_model, f"Dimension mismatch: Input has {C}, but model expects {self.d_model}"
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k_t = k.transpose(-1, -2)
        scores = (q @ k_t) / (self.d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        weights = scores.softmax(axis=-1)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(out)

    def parameters(self):
        return self.w_q.parameters() + self.w_k.parameters() + \
               self.w_v.parameters() + self.w_o.parameters()
