from src.model.core import Value

class no_grad:
    """_summary_
    """
    def __enter__(self):
        self.prev_state = Value.grad_enabled
        Value.grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        Value.grad_enabled = self.prev_state

class precision:
    """Définit la précision des calculs (ex: 32 ou 64 bits)."""
    dtype = "float32"
    def __init__(self, type_str):
        self.type_str = type_str
    def __enter__(self):
        self.old_type = precision.dtype
        precision.dtype = self.type_str
    def __exit__(self, *args):
        precision.dtype = self.old_type