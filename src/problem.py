"""
Generic problem solvable by training a quantum circuit.
"""

import abc


class Problem(metaclass=abc.ABCMeta):
    def __init__(self):
        """
        Initializes the problem object.

        Attributes:
            map (DomainMap): A domain map for the problem.
            target (array): List of taret values for the problem (the true solution of the problem).
            f (callable): A function that returns the true solution value when called for all points in `x`.
        """
        self.map = None
        self.target = None
        self.f = None

    @abc.abstractmethod
    def cost(self, node, weights, data, N):
        """
        Returns the cost function of the problem.

        Parameters:
          node (QuantumNode): The quantum node used for evaluating the circuit.
          weights (array): The array of weights representing the free parameters in the circuit.
          data (array): The input domain data of the problem.
          N (int): The size of the input domain data.
        """
        pass

    @abc.abstractmethod
    def get_domain_map(self):
        """
        Returns:
        (DomainMap): Returns the domain map of the problem.
        """
        pass
