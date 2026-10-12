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

import pennylane as qml
from pennylane import numpy as np

from dataclasses import dataclass


@dataclass
class QiskitPooling:
    """See https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html#2.2-Pooling-Layer"""

    name: str = "Qiskit"
    ppb: int = 3
    nqubits: int = 2

    def layer(p, w):
        qml.RZ(-np.pi / 2, wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[2], wires=w[1])
