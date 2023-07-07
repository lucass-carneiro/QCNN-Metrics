import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def new_QCNN_circuit(num_qubits, npc, npp, ccaf, pcaf, barriers=False):
    """ Creates a new convolutional neural network (CNN) circuit

    Inputs:
    - num_qubits (int): The total number of qubits in the NN. Must be a power of 2
    - npc (int): The number of parameters per convolutional circuit.
    - npp (int): The number of parameters per pooling circuit.
    - barriers (bool): Wether or not to insert barriers between layer blocks. Default: True

    Returns:
    - qiskit.QuantumCircuit: The CNN circuit.
    """

    # We require the number of qubits to be a power of 2. This facilitates reasoning during circuit construction
    # To test this, we use the good old bit twidling hack
    if num_qubits == 0 or not (num_qubits & (num_qubits - 1)) == 0:
        raise ValueError(
            "num_qubits is expected to be a power of 2 and grater than 0")

    # Number of convolutional / pooling layer pairs
    num_layer_pairs = int(np.log2(num_qubits))

    # The circuit with the final network
    qcnn_circ = QuantumCircuit(num_qubits, name="QCNN")

    # The index of the last qubit
    q_last = num_qubits - 1

    for i in range(num_layer_pairs):
        num_pool_prev = num_qubits - int(num_qubits / 2**i)
        num_pool_curr = int(num_qubits / 2**(i + 1))
        append_convolutional_layer(qcnn_circ, num_pool_prev, q_last, "c" + str(i), npc, ccaf, barriers)
        append_pooling_layer(qcnn_circ, num_pool_prev, q_last, num_pool_curr, "p" + str(i), npp, pcaf, barriers)

    return qcnn_circ


def append_convolutional_layer(qcnn_circ, q_start, q_end, param_prefix, param_count, ccaf, barriers):
    """ Appends a convolutional layer on a NN.
    Based on https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html

    Inputs:
    - qnn_circ (qiskit.QuantumCircuit): The NN circuit to append to.
    - q_start (int): Index of the first bit in the layer.
    - q_end (int): Index of the last bit in the layer.
    - param_prefix (str): The name of the parameters of the layer.
    - param_count (int): Number of parameters in the CCAF.
    - ccaf (function): Convolutional Circuit Appending Function (CCAF). The function that models the convolutional circuit.
    - barriers (bool): Wether or not to draw barriers in the blocks.
    """
    layer_size = q_end - q_start + 1

    # The parameter verctor for this layer
    params = ParameterVector(param_prefix, length=(layer_size + 1) * param_count)
    pi = 0

    # Loops over even qubit pairs (01, 23, 45, etc)
    for i in range(q_start, q_end + 1, 2):
        param_slice = slice(pi * param_count, (pi + 1) * param_count)

        ccaf(qcnn_circ, i, i + 1, params[param_slice])

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

    # Place circular boundary
    param_slice = slice(pi * param_count, (pi + 1) * param_count)
    ccaf(qcnn_circ, q_end, q_start, params[param_slice])

    if barriers:
        qcnn_circ.barrier()

    pi += 1


def append_pooling_layer(qcnn_circ, q_start, q_end, num_pooled, param_prefix, param_count, pcaf, barriers):
    """ Appends a pooling layer on a NN.
    Based on https://qiskit.org/ecosystem/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html

    Inputs:
    - qnn_circ (qiskit.QuantumCircuit): The NN circuit to append to.
    - q_start (int): Index of the first bit in the layer.
    - q_end (int): Index of the last bit in the layer.
    - num_pooled (int): Number of quibits that will be pooled (measured) in the layer.
    - param_prefix (str): The name of the parameters of the layer.
    - param_count (int): Number of parameters in the PCAF.
    - pcaf (function): Pooling Circuit Appending Function (PCAF). The function that models the pooling circuit.
    - barriers (bool): Wether or not to draw barriers in the blocks.
    """
    layer_size = q_end - q_start + 1

    if layer_size < num_pooled:
        raise ValueError("The size of the pooling layer is smaller than the number of qubits to pool")

    # The parameter verctor for this layer
    params = ParameterVector(param_prefix, length=num_pooled * param_count)
    pi = 0

    # The index of the quibit that is the first pooling target
    p_start = q_start + num_pooled

    # Polling circuits
    for i in range(q_start, p_start):
        param_slice = slice(pi * param_count, (pi + 1) * param_count)
        pcaf(qcnn_circ, i, num_pooled + i, params[param_slice])

        if barriers:
            qcnn_circ.barrier()

        pi += 1
