import math
class Optimizer:
    """_summary_
    """
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        """_summary_
        """
        for p in self.params:
            p.grad = 0

    def step(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("L'optimizer doit implémenter la méthode step()")

class SGD(Optimizer):
    """_summary_

    Args:
        Optimizer (_type_): _description_
    """
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Adam(Optimizer):
    """_summary_

    Args:
        Optimizer (_type_): _description_
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """_summary_

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 0.001.
            betas (tuple, optional): _description_. Defaults to (0.9, 0.999).
            eps (_type_, optional): _description_. Defaults to 1e-8.
        """
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = [0 for _ in self.params] # moment set to 0
        self.v = [0 for _ in self.params]

    def step(self):
        """_summary_
        """
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
