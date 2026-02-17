from src.tensor_model.nn import Module
from src.tensor_model.nn import MultiHeadAttention
from src.tensor_model.mlp import MLP
from src.tensor_model.nn import LayerNorm

class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len=max_seq_len)
        self.feed_forward = MLP(n_embd=d_model)
        self.ln = LayerNorm(dim=d_model)
    def __call__(self, x):
        
        x = x + self.attention(self.ln(x))
        x = x + self.feed_forward(x) # layernorm in the first layer of feedforward
        return x
    
    def parameters(self):
        return self.attention.parameters() + self.feed_forward.parameters() + self.ln.parameters()