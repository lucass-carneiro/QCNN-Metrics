"""
Model 0:
"""

import utils
from convolutional_layers import vatan_williams

import pennylane as qml
from pennylane import numpy as np

import os


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    y = np.random.rand(1) * x

    A = 1.0 / np.sqrt(np.dot(x, x))
    B = 1.0 / np.sqrt(np.dot(y, y))

    x = A * x
    y = B * y

    return x, y


def new_qcnn(num_qubits):
    def qcnn(p):
        qubit_range = range(num_qubits + 1)
        for previous, current in zip(qubit_range, qubit_range[1:]):
            vatan_williams(p, [previous % num_qubits, current % num_qubits])
    return qcnn


def new_ansatz(num_qubits, qcnn):
    def ansatz(x, p):
        qml.AmplitudeEmbedding(
            features=x, wires=range(num_qubits), normalize=True)
        qcnn(p)
    return ansatz


def cost_hamiltonian(dataset_size, y):
    ket_y = np.reshape(y, (dataset_size, 1))
    bra_y = np.transpose(ket_y)

    ket_y_bra_y = np.dot(ket_y, bra_y)
    identity = np.identity(dataset_size)

    H = identity - ket_y_bra_y

    return qml.pauli_decompose(H)


def train(qcnn, ansatz, x, y, H, q_device, num_qubits, num_params, arguments):
    plot_cost = bool(arguments["--plot-cost"])
    fisher_samples = int(arguments["--fisher-samples"])

    # Optimize
    i, cost_data, params = utils.optimize(
        ansatz,
        x,
        H,
        q_device,
        num_params,
        arguments
    )

    # Plot cost
    if plot_cost:
        if not os.path.exists("img/model_0/"):
            os.makedirs("img/model_0/")
        utils.plot_cost(cost_data, i + 1, "img/model_0/cost.pdf")

    # Fisher matrices
    cfm = utils.classical_fisher(
        ansatz, x, q_device, num_params, fisher_samples)
    qfm = utils.quantum_fisher(qcnn, num_params, num_qubits, fisher_samples)

    # Save data
    if not os.path.exists("data/"):
        os.makedirs("data/")

    utils.save_data("data/model_0.hdf5", i, cost_data, params, x, y, cfm, qfm)


def draw(qcnn, q_device, ansatz, dataset_size, num_qubits):
    if not os.path.exists("img/model_0/"):
        os.makedirs("img/model_0/")

    utils.draw(qcnn, q_device, "img/model_0/qcnn.pdf",
               [0.0] * 3 * num_qubits)

    utils.draw(ansatz, q_device, "img/model_0/ansatz.pdf",
               [1.0] * dataset_size, [0.0] * 3 * num_qubits)


def main(arguments):
    if arguments["fisher-spectrum"]:
        utils.compute_fisher_spectrum("data/model_0.hdf5")
        return

    dataset_size = int(arguments["<dataset-size>"])
    assert dataset_size > 0

    # Normalized dataset
    x, y = new_dataset(-1.0, 1.0, dataset_size)

    # Simulator
    q_device, num_qubits = utils.new_device(dataset_size)

    # VQC
    qcnn = new_qcnn(num_qubits)
    num_params = 3 * num_qubits

    # Ansatz
    ansatz = new_ansatz(num_qubits, qcnn)

    # Cost Hamiltonian
    H = cost_hamiltonian(dataset_size, y)

    if arguments["train"]:
        train(
            qcnn,
            ansatz,
            x,
            y,
            H,
            q_device,
            num_qubits,
            num_params,
            arguments,
        )
        return

    if arguments["draw"]:
        draw(qcnn, q_device, ansatz, dataset_size, num_qubits)
        return
