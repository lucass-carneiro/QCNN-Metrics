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
Contains definitions for taking derivatives of quantum circuits that represent
functions.
"""
from pennylane import numpy as np


def df(node, weights, x):
    """
    Takes the first derivative of a quantum circuit.

    Parameters:
      node (QuantumNode): The quantum node that will evaluate the circuit.
      weights (array): A list of circuit weights.
      x (float): The point where the derivative is to be computed.

    Returns:
      (flot): The value of the first derivative of the function represented by the circuit in point `x`
    """
    fp_2 = node(weights, x=(x + np.pi / 2.0))
    fm_2 = node(weights, x=(x - np.pi / 2.0))
    return (fp_2 + fm_2) / 2.0


def d2f(node, weights, x):
    """
    Takes the second derivative of a quantum circuit.

    Parameters:
      node (QuantumNode): The quantum node that will evaluate the circuit.
      weights (array): A list of circuit weights.
      x (float): The point where the derivative is to be computed.

    Returns:
      (flot): The value of the second derivative of the function represented by the circuit in point `x`
    """
    f = node(weights, x=x)
    fp = node(weights, x=(x + np.pi))
    fm = node(weights, x=(x - np.pi))
    return (fm + fp - 2.0 * f) / 4.0
