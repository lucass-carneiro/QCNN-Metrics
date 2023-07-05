from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def new_convolutional_layer(num_qubits, param_prefix, param_count, layer_function):
    """Builds a convolutional layer cirquit of any number of qubits.

    Parameters:
    num_quibits (int): The number of quibits in the layer.
    param_prefix (string): The prefix (symbol) used to identify parameters in the circuit.
    param_count (int): Number of parameters in the layer block function
    layer_function (function): A function that returns a 2-quibit parametric circuit that will be used in the layer.

    Returns:
    qiskit.QuantumCircuit: The convolutional layer circuit
    """
    qc = QuantumCircuit(num_qubits, name = "Convolutional Layer")

    # TODO: Is this early bailling out correct?
    if num_qubits == 2:
        params = ParameterVector(param_prefix, length = param_count)
        qc.compose(layer_function(params[0:param_count]), inplace = True)
        qc.barrier()
        return qc

    params = ParameterVector(param_prefix, length = num_qubits * param_count)

    # Iterate over pairs from bottom to top
    for i in range(num_qubits):
        param_slice = slice(i * param_count, (i + 1) * param_count)
        qc.compose(layer_function(params[param_slice]), [i, (i + 1) % num_qubits], inplace = True)
        qc.barrier()

    return qc
