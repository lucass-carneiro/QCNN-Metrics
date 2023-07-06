import numpy as np


def vatan_williams(qc, q0, q1, params):
    """Appends the generalized 2-qubit unitary defined in https://arxiv.org/abs/quant-ph/0308006
    to an existing circuit

    Parameters:
    qc (qiskit.QuantumCircuit): The circuit to append to.
    q0 (int): The first qubit in the circuit.
    q1 (int): The second qubit in the circuit.
    params (qiskit.ParameterVector): Circuit parameters.
    """

    qc.rz(-np.pi / 2, q1)
    qc.cnot(q1, q0)
    qc.rz(params[0], q0)
    qc.ry(params[1], q1)
    qc.cx(q0, q1)
    qc.ry(params[2], q1)
    qc.cx(q1, q0)
    qc.rz(np.pi / 2, q0)


def qiskit_pooling(qc, q0, q1, params):
    """Appennd the 2-qubit unitary for pooling used in the qiskit tutorials
    https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html#2.2-Pooling-Layer

    Parameters:
    qc (qiskit.QuantumCircuit): The circuit to append to.
    q0 (int): The first qubit in the circuit.
    q1 (int): The second qubit in the circuit.
    params (qiskit.ParameterVector): Circuit parameters.
    """

    qc.rz(-np.pi / 2, q1)
    qc.cnot(q1, q0)
    qc.rz(params[0], q0)
    qc.ry(params[1], q1)
    qc.cnot(q0, q1)
    qc.ry(params[2], q1)

    return qc
