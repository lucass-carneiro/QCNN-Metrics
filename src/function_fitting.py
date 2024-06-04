import quantum_derivatives as qd
import domain_map as dm

import pennylane.numpy as np


class FitToFunction:
    def __init__(self, x0: float, xf: float, f):
        self.map = dm.LinearMap(x0, xf)
        self.f = f

    def cost_pointwise(self, node, weights, x):
        X = self.map.global2local(x)
        Y = node(weights, x=X)
        y = self.f(x)

        return (Y - y)

    def cost(self, node, weights, data, N):
        s = sum(self.cost_pointwise(node, weights, x) ** 2 for x in data)
        return s / N
