import numpy as np
from qiskit import QuantumCircuit

def vatan_williams(params):
    """Generalized 2-quibit unitary defined in https://arxiv.org/abs/quant-ph/0308006

    Parameters:
    params (qiskit.ParameterVector): Circuit parameters

    Returns:
    qiskit.QuantumCircuit: The parametric circuit block
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cnot(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc
