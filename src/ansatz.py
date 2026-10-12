# QCNN Toolkit - training Quantum Convolutional Neural Networks for
# fitting functions and solving differential equations.
# Copyright (C) 2023-2024  Lucas Timotheo Sanches, Fabio Pereira dos Santos
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
