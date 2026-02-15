import numpy as np
from src.tensor_model.tensor import Tensor

class DataLoader:
    """_summary_
    """
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]

    def __iter__(self):
        """_summary_

        Yields:
            _type_: _description_
        """
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield Tensor(self.X[batch_idx]), Tensor(self.y[batch_idx])

