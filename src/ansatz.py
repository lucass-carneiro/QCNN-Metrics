"""
This module defines an abstract `anstz` circuit.
An `ansatz` circuit is the combination of training and encoding blocks
as defined [here](https://arxiv.org/abs/2008.08605)
"""

import abc


class Ansatz(metaclass=abc.ABCMeta):
    """
    Abstract class defining an `ansatz` quantum circuit that combines
    encoding and training blocks. See [here](https://arxiv.org/abs/2008.08605)
    for details
    """

    @abc.abstractmethod
    def S(self, x):
        """
        The enconder block for the ansatz circuit.

        Parameters:
          x (array): An array-like object containing the datapoints to encode.
        """
        pass

    @abc.abstractmethod
    def W(self, theta):
        """
        The trainable block for the ansatz circuit.

        Parameters:
          theta (array): An array-like object containing the list of trainable parameters.
        """
        pass

    @abc.abstractmethod
    def ansatz(self, w, x):
        """
        The ansatz circuit containing enconding and trainable blocks.

        Parameters:
          w (array): An object containing the list of trainable parameters.
          x (array): An object containing the list of datapoints to encode.

        Returns:
          (pennylane.ExpectationMP): The expected value of the quantum circuit
        """
        pass
