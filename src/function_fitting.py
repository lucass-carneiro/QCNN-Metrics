"""
A problem type defined by training a quantum circuit to fit a target function.
"""

import problem as prb
import domain_map as dm

import pennylane.numpy as np

import torch


def f_torch(x):
    """
    The target function, using the `torch` library.

    Parameters:
     x (float): the point where to evaluate the function.
    """
    return torch.exp(x * torch.cos(3.0 * torch.pi * x)) / 2.0


def f_numpy(x):
    """
    The target function, using the `torch` library.

    Parameters:
     x (float): the point where to evaluate the function.
    """
    return np.exp(x * np.cos(3.0 * np.pi * x)) / 2.0


class FitToFunction(prb.Problem):
    """
    Defines a function fitting problem for the quantum circuit to solve.
    """

    def __init__(self, x0: float, xf: float, x, optimizer: str):
        """
        Initializes the problem object

        Parameters:
        x0 (float): Left boundary of the domain.
        xf (float): Righr boundary of the domain.
        x (array): List of input points in the[x0, xf] range
        optimizer (str): Name of the optimizer library to use.
        """
        self.map = dm.LinearMap(x0, xf)
        self.target = f_numpy(x)

        if optimizer == "numpy":
            self.f = f_numpy
        elif optimizer == "torch":
            self.f = f_torch
        else:
            print(f"Unrecognized optimizer \"{optimizer}\"")
            exit(1)

    def get_domain_map(self):
        """
        Returns:
         (DomainMap): The linear domain map used in the problem.
        """
        return self.map

    def cost_pointwise(self, node, weights, x):
        """
        Returns the cost function of the problem at each point in the
        input domain.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          x (float): The point where to comput the cost function.
        """
        X = self.map.global2local(x)
        Y = node(weights, x=X)
        y = self.f(x)

        return (Y - y)

    def cost(self, node, weights, data, N):
        """
        Returns the cost function of the problem.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          data (array): The input domain data of the problem.
          N (int): The size of the input domain data.
        """
        s = sum(self.cost_pointwise(node, weights, x) ** 2 for x in data)
        return s / N
