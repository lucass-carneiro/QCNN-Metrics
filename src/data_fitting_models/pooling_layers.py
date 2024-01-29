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
