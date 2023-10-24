"""
Model 2:
"""

import utils
from convolutional_layers import hur_kim_park_5 as unitary_block_function
from convolutional_layers import hur_kim_park_5_ppb as param_per_block

import pennylane as qml
from pennylane import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os

model_name = "model_2"
img_folder = os.path.join("img", model_name)
model_data_file = os.path.join("data", model_name + ".hdf5")


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    y = 0.3 * x + 0.5

    return x, y


def encode(num_qubits, data):
    assert (len(data) == num_qubits)

    for i in range(num_qubits):
        qml.RY(data[i], i)


def new_qcnn(num_qubits):
    def qcnn(p):
        qubit_range = range(num_qubits + 1)
        for previous, current in zip(qubit_range, qubit_range[1:]):
            unitary_block_function(
                p, [previous % num_qubits, current % num_qubits])
    return qcnn


def new_ansatz(num_qubits, qcnn):
    def ansatz(x, p):
        encode(num_qubits, x)
        qcnn(p)
    return ansatz


def optimize(ansatz, x, y, q_device, num_qubits, num_params, arguments):
    max_iters = int(arguments["--max-iters"])
    abstol = float(arguments["--abstol"])

    # Cost and cost node
    def cost_circuit(p):
        ansatz(x, p)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    cost_node = qml.QNode(cost_circuit, q_device, interface="autograd")

    N = len(y)

    def cost(p):
        xi = np.arccos(cost_node(p))/2
        return np.sqrt(np.sum((y - xi)**2) / N)

    # Optimization parameters and cost vector
    cost_data = []
    params = np.random.normal(-np.pi, np.pi, num_params, requires_grad=True)

    # Optimize
    opt = qml.SPSAOptimizer(maxiter=max_iters)

    print("Optimizing")

    stopping_criteria = "max iterations reached"

    for i in range(max_iters):
        params, loss = opt.step_and_cost(cost, params)
        cost_data.append(loss)
        print("Loss in teration", i, "=", loss)

        if np.abs(loss) < abstol:
            stopping_criteria = "absolute tolerance reached"
            break

    # Results
    print("Training results:")
    print("  Stopping criteria: ", stopping_criteria)
    print("  Iterations:", i + 1)
    print("  Final cost value:", cost_data[-1])
    print("  Final training parameters:", params)

    return i, cost_data, params


def train(qcnn, ansatz, x, y, q_device, num_qubits, num_params, arguments):
    plot_cost = bool(arguments["--plot-cost"])
    fisher_samples = int(arguments["--fisher-samples"])

    # Optimize
    i, cost_data, params = optimize(
        ansatz,
        x,
        y,
        q_device,
        num_qubits,
        num_params,
        arguments
    )

    # Plot cost
    if plot_cost:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        utils.plot_cost(cost_data, i + 1, os.path.join(img_folder, "cost.pdf"))

    # Fisher matrices
    cfm = utils.classical_fisher(qcnn, num_params, num_qubits, fisher_samples)
    qfm = utils.quantum_fisher(qcnn, num_params, num_qubits, fisher_samples)

    # Save data
    if not os.path.exists("data"):
        os.makedirs("data")

    utils.save_data(model_data_file, i, cost_data, params, x, y, cfm, qfm)


def draw(qcnn, q_device, ansatz, dataset_size, num_qubits):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    utils.draw(qcnn, q_device, os.path.join(img_folder, "qcnn.pdf"),
               [0.0] * param_per_block * num_qubits)

    utils.draw(ansatz, q_device, os.path.join(img_folder, "ansatz.pdf"),
               [1.0] * dataset_size, [0.0] * param_per_block * num_qubits)


def validate(data_file, ansatz, q_device, num_qubits, plot_file, font_size=18):
    with h5py.File(data_file, "r") as f:
        x = np.array(f.get("trainig_data/training_set"))
        y = np.array(f.get("trainig_data/target_set"))
        p = np.array(f.get("trainig_data/params"))

    # QCNN state and node
    def ansatz_func(P):
        ansatz(x, P)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    ansatz_node = qml.QNode(ansatz_func, q_device)
    x_theta = np.arccos(ansatz_node(p)) / 2
    print("Original data")
    print(y)
    print("Trained data")
    print(x_theta)
    print("Error")
    error = np.abs(np.abs(y) - np.abs(x_theta))
    print(error)

    mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size

    plt.close("all")

    f, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(box_aspect=1))

    # Fit
    ax1.scatter(x, y, label="Input data", color="black")
    ax1.scatter(x, x_theta, label="Trained", color="red")

    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend()

    ax1.set_xlabel("$x$", fontsize=font_size)
    ax1.set_ylabel("$y$", fontsize=font_size)

    # Error
    ax2.scatter(x, error, color="black")

    ax1.set_aspect("equal", adjustable="datalim")

    ax2.set_xlabel("$x$", fontsize=font_size)
    ax2.set_ylabel("Error", fontsize=font_size)

    f.tight_layout()
    f.savefig(plot_file, bbox_inches="tight")


def main(arguments):
    if arguments["archive"]:
        utils.archive(img_folder, model_data_file, model_name)
        return

    if arguments["fisher-spectrum"]:
        utils.plot_fisher_spectrum(
            model_data_file,
            os.path.join(img_folder, "spectrum.pdf"),
            arguments["--quantum"]
        )
        return

    if arguments["validate"]:
        dataset_size = utils.get_dataset_size(model_data_file)
    else:
        dataset_size = int(arguments["<dataset-size>"])

    assert dataset_size > 0

    # Normalized dataset
    x, y = new_dataset(-1.0, 1.0, dataset_size)

    # Simulator
    num_qubits = len(x)
    q_device = qml.device("default.qubit", wires=num_qubits, shots=None)

    # VQC
    qcnn = new_qcnn(num_qubits)
    num_params = param_per_block * num_qubits

    # Ansatz
    ansatz = new_ansatz(num_qubits, qcnn)

    if arguments["train"]:
        train(
            qcnn,
            ansatz,
            x,
            y,
            q_device,
            num_qubits,
            num_params,
            arguments,
        )
        return

    if arguments["draw"]:
        draw(qcnn, q_device, ansatz, dataset_size, num_qubits)
        return

    if arguments["validate"]:
        validate(
            model_data_file,
            ansatz,
            q_device,
            num_qubits,
            os.path.join(img_folder, "validation.pdf")
        )
        return
