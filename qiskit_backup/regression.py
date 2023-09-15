import numpy as np

from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit import execute

import multiprocessing as mproc

import h5py

import unitary_blocks as ubs

sim_shots = 1000


def amplitude_encode(data_array, num_qubits, block_label):
    """ Encodes a numpy array into a quantum circuit using amplitude encoding.

    Inputs:
    - data_array (numpy array): The data to encode.
    - num_qubits (int): The number of qubits used to encode the data.
    - block_label (str): The label to give to the unitary block.

    Returns:
    - qiskit.extensions.UnitaryGate: The state preparation unitary gate with the data encoded in the circuit
    """
    norm_factor = 1.0 / np.linalg.norm(data_array)

    normalized_data = norm_factor * data_array

    V = QuantumCircuit(num_qubits)
    V.prepare_state(normalized_data)

    return UnitaryGate(Operator(V), label=block_label)


def make_ansatz(vqc, x_data, draw=True):
    num_qubits = int(np.ceil(np.log2(len(x_data))))

    assert num_qubits == vqc.num_qubits

    U_x = amplitude_encode(x_data, num_qubits, "U_x")

    qubit_range = [*range(0, num_qubits)]

    ansatz = QuantumCircuit(num_qubits)
    ansatz.append(U_x, qubit_range)
    ansatz.append(vqc.to_gate(label="VQC"), qubit_range)

    if draw:
        ansatz.draw(output="mpl", filename="img/ansatz.svg")

    return ansatz


def make_cost_operator(ansatz, y_data, draw=True):
    num_qubits = int(np.ceil(np.log2(len(y_data))))

    assert num_qubits == ansatz.num_qubits

    S_y_dagg = amplitude_encode(y_data, num_qubits, "S_y").adjoint()
    S_y_dagg._label = "S_y_dagg"

    qubit_range = [*range(0, num_qubits)]

    cost_op = QuantumCircuit(num_qubits)
    cost_op.append(ansatz.to_gate(label="Ansatz"), qubit_range)
    cost_op.append(S_y_dagg, qubit_range)

    if draw:
        cost_op.draw(output="mpl", filename="img/cost_operator.svg")

    return cost_op


def add_hadamard_test(cost_op, draw=True):
    test_qubits = cost_op.num_qubits + 1

    cost_op_gate = cost_op.to_gate(label="Cost OP").control(1)

    H_test = QuantumCircuit(test_qubits)
    H_test.h(0)
    H_test.append(cost_op_gate, [*range(0, test_qubits)])
    H_test.control()
    H_test.h(0)

    if draw:
        H_test.draw(output="mpl", filename="img/hadamard_test.svg")

    return H_test


def eval_pqc(pqc, sim, params):
    # https://quantumcomputing.stackexchange.com/a/7131
    pqc_with_params = pqc.assign_parameters(params)

    job = execute(pqc_with_params, sim, shots=sim_shots)
    sv = np.array(job.result().get_statevector(pqc_with_params))
    return sv


def Hadamard_test(H_test_qc, sim, params):
    sv = eval_pqc(H_test_qc, sim, params)

    P0 = np.abs(sv[0])**2
    P1 = np.abs(sv[1])**2

    return P0 - P1


def global_cost(H_test_qc, sim, params):
    return 1.0 - np.abs(Hadamard_test(H_test_qc, sim, params))**2


def global_cost_with_params(H_test_qc, sim, params):
    return [*params, global_cost(H_test_qc, sim, params)]


def parallel_sweep_global_cost(a_end, a_count, dims, H_test_qc, sim):
    # Create list of parameters
    assert a_end > 0

    a_start = -a_end
    delta_a = (a_end - a_start) / (a_count - 1)

    a_0 = np.array([a_start] * dims)

    param_list = [
        (
            H_test_qc,
            sim,
            a_0 + np.array(I) * delta_a,
        )
        for I in np.ndindex((a_count,) * dims)
    ]

    # Distribute work and sore furues in queue
    pool = mproc.get_context("spawn").Pool(processes=mproc.cpu_count())

    result = pool.starmap_async(global_cost_with_params, param_list)

    pool.close()
    pool.join()

    return result.get()


def dump_cost(a_end, a_count, dims, H_test_qc, sim):
    cost_data = parallel_sweep_global_cost(
        a_end,
        a_count,
        dims,
        H_test_qc,
        sim
    )

    with h5py.File("data/cost_data.hdf5", "w") as f:
        f.create_dataset("cost_data", dtype=float, data=cost_data)


def save_fit(x_data, y_data, ansatz, sim, params):
    ansatz_eval = np.real(eval_pqc(ansatz, sim, params))
    y_norm = y_data / np.linalg.norm(y_data)

    with h5py.File("data/fit_data.hdf5", "w") as f:
        f.create_dataset("input_data", dtype=float, data=x_data)
        f.create_dataset("normalized_target_data", dtype=float, data=y_norm)
        f.create_dataset("fit_params", dtype=float, data=params)
        f.create_dataset("fitted_data", dtype=float, data=ansatz_eval)
