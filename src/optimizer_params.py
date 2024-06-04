class OptimizerParams:
    def __init__(self, max_iters: int, abstol: float, step: float, batch_size: int):
        self.max_iters = max_iters
        self.abstol = abstol
        self.step = step
        self.batch_size = batch_size