from src.model.core import Value
# pylint: disable=protected-access
class NoGrad:
    """_summary_
    """
    def __enter__(self):
        """_summary_
        """
        self.prev_state = Value.grad_enabled
        Value.grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """_summary_

        Args:
            exc_type (_type_): _description_
            exc_val (_type_): _description_
            exc_tb (_type_): _description_
        """
        Value.grad_enabled = self.prev_state

