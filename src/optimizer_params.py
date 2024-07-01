"""
Contains optimizer related configurations
"""


class OptimizerParams:
    """
    A databag for holding circuit optimizer related data.

    Attributes:
        max_iters (float): The maximun number of steps the optimizer may take.
        abstol (float): The absolute tolerence of the error function below which training stops.
        step (float): Initial step size for the optimizer algorithim.
        batch_size (int): Size of the subdata array to use for batching. If `0`, batching is disabled.
    """

    def __init__(self, max_iters: int, abstol: float, step: float, batch_size: int):
        """
        Initialize the object.

        Parameters:
            max_iters (float): The maximun number of steps the optimizer may take.
            abstol (float): The absolute tolerence of the error function below which training stops.
            step (float): Initial step size for the optimizer algorithim.
            batch_size (int): Size of the subdata array to use for batching. If `0`, batching is disabled.
        """
        self.max_iters = max_iters
        self.abstol = abstol
        self.step = step
        self.batch_size = batch_size
