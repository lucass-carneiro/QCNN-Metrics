import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def new_CNN(num_qubits, ccaf):

    # We require the number of qubits to be even. This facilitates reasoning during circuit construction
    # To test if the number is a odd, we use the good old bit twidling hack
    if num_qubits == 0 or num_qubits & 1:
        raise ValueError(
            "num_qubits is expected to be even and grater than 0")

    # Number of convolutional / pooling layer pairs
    num_layer_pairs = 0  # TODO

    # The circuit with the final network
    qcnn_circ = QuantumCircuit(num_qubits, name="QCNN")

    # Convolutional / Pooling pair sequences construction
    append_convolutional_layer(qcnn_circ, 0, 7, "c1", 3, ccaf)

    return qcnn_circ


# ccaf = convolutional circuit appending function
# pcaf = pooling circuit appending function

# https://arxiv.org/abs/1810.03787
def append_convolutional_layer(qcnn_circ, q_start, q_end, param_prefix, param_count, ccaf, barriers=True):
    size = q_end - q_start + 1

    # The parameter verctor for this layer
    params = ParameterVector(param_prefix, length=(size + 1) * param_count)
    pi = 0

    # Upper boundary
    param_slice = slice(pi * param_count, (pi + 1) * param_count)
    ccaf(qcnn_circ, q_end, q_start, params[param_slice])

    if barriers:
        qcnn_circ.barrier()

    pi += 1

    # Loops over odd qubit pairs (12, 34, 56, etc)
    for i in range(q_start + 1, q_end, 2):
        param_slice = slice(pi * param_count, (pi + 1) * param_count)

        ccaf(qcnn_circ, i, i + 1, params[param_slice])
        if barriers:
            qcnn_circ.barrier()

        pi += 1

    # Lower boundary
    param_slice = slice(pi * param_count, (pi + 1) * param_count)
    ccaf(qcnn_circ, q_start, q_end, params[param_slice])

    if barriers:
        qcnn_circ.barrier()

    pi += 1

    # Loops over even qubit pairs (01, 23, 45, etc)
    for i in range(q_start, q_end + 1, 2):
        param_slice = slice(pi * param_count, (pi + 1) * param_count)

        ccaf(qcnn_circ, i, i + 1, params[param_slice])

        if barriers:
            qcnn_circ.barrier()

        pi += 1


# indicies of qubits that need to be eliminated
# def append_pooling_layer(qcnn_circ, q_start, q_end, param_prefix, param_count, pcaf, barriers=True):
