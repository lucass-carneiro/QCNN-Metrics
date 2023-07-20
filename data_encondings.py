import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap


def amplitude_encode(*args):
    if len(args) == 0:
        raise "Cannot encode empty data list"

    alpha = np.concatenate(args)
    A = 1.0 / np.linalg.norm(alpha)

    return A, alpha, int(np.log2(len(alpha)))
