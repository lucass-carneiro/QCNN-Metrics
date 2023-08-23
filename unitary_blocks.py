import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


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


def new_vatan_williams(draw=True):
    qc = QuantumCircuit(2)
    vatan_williams(qc, 0, 1, ParameterVector("p", 3))
    qc.draw(output="mpl", filename="vqc.svg")
    return qc


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


def vatan_williams_2_param(qc, q0, q1, p2, params):
    """Appends the generalized 2-qubit unitary defined in https://arxiv.org/abs/quant-ph/0308006
    to an existing circuit, but excluding the last parametrized entangling block

    Parameters:
    qc (qiskit.QuantumCircuit): The circuit to append to.
    q0 (int): The first qubit in the circuit.
    q1 (int): The second qubit in the circuit.
    p2 (int): The index of the parameter to repeat for the last gate
    params (qiskit.ParameterVector): Circuit parameters.
    """

    qc.rz(-np.pi / 2, q1)
    qc.cnot(q1, q0)
    qc.rz(params[0], q0)
    qc.ry(params[1], q1)
    qc.cx(q0, q1)
    qc.ry(params[p2], q1)
    qc.cx(q1, q0)
    qc.rz(np.pi / 2, q0)


def new_vatan_williams_2_param(p2, draw=True):
    qc = QuantumCircuit(2)
    vatan_williams_2_param(qc, 0, 1, p2, ParameterVector("p", 2))
    qc.draw(output="mpl", filename="vqc.svg")
    return qc
