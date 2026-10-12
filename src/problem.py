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
