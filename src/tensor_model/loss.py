import numpy as np
from src.tensor_model.tensor import Tensor

def cross_entropy(logits, targets):
    probs = logits.softmax()
    batch_size = probs.data.shape[0]
    correct_logprobs = probs.data[np.arange(batch_size), targets] 
    loss_data = -np.log(correct_logprobs + 1e-15)
    out = Tensor(np.mean(loss_data), (probs,), 'cross_entropy')
    def _backward():
        d_logits = probs.data.copy() 
        d_logits[np.arange(batch_size), targets] -= 1 # p-y
        d_logits /= batch_size
        probs._parent_grad_update(out.grad * d_logits)
    out._backward = _backward