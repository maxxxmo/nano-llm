import numpy as np
from src.tensor_model.utils import unbroadcast

class Tensor ():
    def __init__ (self, data, _children=(), _op=''):
        self.data = np.array(data)
        self._children = _children # c = a + b, c is children a & b are prev
        self.op = _op
        self.grad = np.zeros_like(data)
        self._prev = set(_children)
        self._backward = lambda: None
    def __repr__ (self):
        return f"Tensor(data={self.data},shape = {self.data.shape}, grad = {self.grad})"
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data) # difference with micrograd
        for node in reversed(topo):
            node._backward()
    
    def __add__ (self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # result is broadcasted by default if tensor are different sizes
        out = Tensor(self.data + other.data, _children=((self, other)), _op='+')
        def _backward():
            # for addition we don't put chain rule as its equals to *
            grad_self = out.grad
            grad_other = out.grad
            # UNBROADCAST + ACCUMULATION
            self.grad += unbroadcast(grad_self, self.data.shape)
            other.grad += unbroadcast(grad_other, other.data.shape)
        out._backward = _backward
        return out
    
    def __mul__ (self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=((self, other)), _op='*')
        def _backward():
            # for addition we don't put chain rule as its equals to *
            grad_self = out.grad * other.data
            grad_other = out.grad * self.data  
            # UNBROADCAST + ACCUMULATION
            self.grad += unbroadcast(grad_self, self.data.shape)
            other.grad += unbroadcast(grad_other, other.data.shape)
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=((self, other)), _op='@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
        
    def softmax(self, axis=-1):
        # axis = -1 --> by line
        max_val = np.max(self.data,axis=axis,keepdims=True)
        exps = np.exp(self.data-max_val)
        norms = np.sum(exps,axis=axis,keepdims=True)
        out = Tensor(exps/norms,(self,),'softmax')
        def _backward():
            self.grad += out.data * (out.grad - np.sum(out.grad * out.data, axis=-1, keepdims=True))
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        def _backward():
            mask = (self.data > 0).astype(float)
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def gelu(self):
        x = self.data
        tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
        out_data = 0.5 * x * (1 + tanh_out)
        out = Tensor(out_data, (self,), 'GeLU')
        def _backward():
            term = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
            pdf = 0.5 * (1 + tanh_out) + (0.5 * x * (1 - tanh_out**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))
            self.grad += pdf * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Seuls les int/float sont supportés"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            local_grad = (other * (self.data**(other - 1))) * out.grad
            self.grad += unbroadcast(local_grad, self.data.shape)
        out._backward = _backward
        return out

    def __neg__(self): # -self
        """_summary_

        Returns:
            _type_: _description_
        """
        return self * -1

    def __radd__(self, other): # other + self
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self + other

    def __sub__(self, other): # self - other
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self + (-other)
 
    def __rsub__(self, other): # other - self
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return other + (-self)

    def __rmul__(self, other): # other * self
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self * other

    def __truediv__(self, other): # self / other
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        """_summary_

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        return other * self**-1

    def __repr__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return f"Value(data={self.data}, grad={self.grad})"

def mean(self, axis=None, keepdims=False):
    out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, (self,), 'mean')
    def _backward():
        n = self.data.size / out.data.size
        grad = out.grad / n
        self.grad += unbroadcast(np.ones_like(self.data) * grad, self.data.shape)
    out._backward = _backward
    return out
