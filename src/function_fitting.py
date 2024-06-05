import problem as prb
import domain_map as dm

import pennylane.numpy as np

import torch


def f_torch(x):
    return torch.exp(x * torch.cos(3.0 * torch.pi * x)) / 2.0


def f_numpy(x):
    return np.exp(x * np.cos(3.0 * np.pi * x)) / 2.0


class FitToFunction(prb.Problem):
    def __init__(self, x0: float, xf: float, optimizer):
        self.map = dm.LinearMap(x0, xf)
        if optimizer == "numpy":
            self.f = f_numpy
        elif optimizer == "torch":
            self.f = f_torch
        else:
            print(f"Unrecognized optimizer \"{optimizer}\"")
            exit(1)

    def cost_pointwise(self, node, weights, x):
        X = self.map.global2local(x)
        Y = node(weights, x=X)
        y = self.f(x)

        return (Y - y)

    def cost(self, node, weights, data, N):
        s = sum(self.cost_pointwise(node, weights, x) ** 2 for x in data)
        return s / N
