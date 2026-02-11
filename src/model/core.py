# Value class constructed on micrograd example of Andrej Karpathy
import numpy as np
# pylint: disable=protected-access
class Value :
    """Value Class to follow operations
    """
    def __init__(self, data, _children =(), _op=''):
    
            """
            init
            Args:
                data (_type_): _description_
                _children (tuple, optional): _description_. Defaults to ().
                _op (str, optional): _description_. Defaults to ''.
            """
            self.data = data
            self.grad = 0
            # For graph
            self._backward = lambda:None  # no gradient at initialisation
            self._prev = set(_children) # pour c = a + b les parents sont {a,b}
            self._op = _op  # Operation that created this value
    
    def __add__(self, other):
        """
        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        Returns:
            _type_: _description_
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        """
        Returns:
            _type_: _description_
        """
        assert self.data > 0, 'log value must be positive (in core)'
        out = Value(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        """
        Returns:
            _type_: _description_
        """
        out = Value(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        """
        Returns:
            _type_: _description_
        """
        out = Value(( 1 - np.exp(2* self.data)) / (1 + np.exp( -2 * self.data)), (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - out.data**2)
        out._backward= _backward
        return out
    
    def sigmoid(self):
        """
        Returns:
            _type_: _description_
        """
        out = Value(np.exp(self.data) / (1 + np.exp(self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward= _backward
        return out
        
    def leaky_relu(self, slope= 0.01): # https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        """
        Args:
            slope (float, optional): _description_. Defaults to 0.01.

        Returns:
            _type_: _description_
        """
        out = Value(slope * self.data if self.data < 0 else self.data, (self,), 'leaky_relu')
        def _backward():
            self.grad +=  slope * out.grad if self.data < 0 else out.grad
        out._backward= _backward
        return out
    

    def backward(self):
        """
        """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)
 
    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    