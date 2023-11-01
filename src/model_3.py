"""
Model 3:
"""

import utils

from convolutional_layers import hur_kim_park_6 as unitary_block_function
from convolutional_layers import hur_kim_park_6_ppb as param_per_block

import pennylane as qml
from pennylane import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os

model_name = "model_3"
img_folder = os.path.join("img", model_name)
model_data_file = os.path.join("data", model_name + ".hdf5")


def new_dataset(dataset_size):
    x = np.linspace(-1.0, 1.0, num=dataset_size)
    x = x / np.sqrt(np.dot(x, x))
    y = x*x*x
    return x, y


def optimize(args, ansatz, device, x, y, dataset_size, num_params):
    max_iters = int(args["--max-iters"])
    abstol = float(args["--abstol"])

    cost_node = qml.QNode(ansatz, device, interface="autograd")
    target = y.astype(complex)

    def cost(p):
        trained = cost_node(p)

        cost_diff = target - trained
        cost_diffS = np.conjugate(cost_diff)

        return np.sum(np.sqrt(np.real(cost_diff * cost_diffS))) / dataset_size

    # Optimization parameters and cost vector
    cost_data = []
    params = np.random.normal(-x[0], x[-1], num_params, requires_grad=True)

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


def train(args, qcnn, ansatz, device, x, y, dataset_size, num_params, num_qubits):
    plot_cost = bool(args["--plot-cost"])
    fisher_samples = int(args["--fisher-samples"])

    # Optimize
    i, cost_data, params = optimize(
        args, ansatz, device, x, y, dataset_size, num_params)

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


def draw(qcnn, ansatz, q_device, num_qubits):
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    utils.draw(qcnn, q_device, os.path.join(img_folder, "qcnn.pdf"),
               [0.0] * param_per_block * num_qubits)

    utils.draw2(ansatz, q_device, os.path.join(img_folder, "ansatz.pdf"),
                [0.0] * param_per_block * num_qubits)


def validate(data_file, ansatz, device, num_qubits, plot_file, font_size=18):
    with h5py.File(data_file, "r") as f:
        x = np.array(f.get("trainig_data/training_set"))
        y = np.array(f.get("trainig_data/target_set"))
        p = np.array(f.get("trainig_data/params"))

    ansatz_node = qml.QNode(ansatz, device)
    x_theta = np.real(ansatz_node(p))
    print("Original data")
    print(y)
    print("Trained data")
    print(x_theta)
    print("Error")
    error = np.abs(np.abs(y) - np.abs(x_theta))
    print(error)

    mpl.rcParams['mathtext.fontset'] = 'cm'
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


def main(args):
    if args["validate"]:
        dataset_size = utils.get_dataset_size(model_data_file)
    else:
        dataset_size = int(args["<dataset-size>"])

    assert dataset_size > 0

    # Data (already normalized)
    x, y = new_dataset(dataset_size)

    # Devices
    num_qubits = int(np.log2(dataset_size))
    device = qml.device("default.qubit", wires=num_qubits, shots=None)

    # VQC
    def qcnn(p):
        qubit_range = range(num_qubits + 1)
        for previous, current in zip(qubit_range, qubit_range[1:]):
            unitary_block_function(
                p, [previous % num_qubits, current % num_qubits])

    num_params = param_per_block * num_qubits

    # Ansatz
    def ansatz(p):
        qml.AmplitudeEmbedding(features=x, wires=range(num_qubits))
        qcnn(p)
        return qml.state()

    if args["train"]:
        train(
            args, qcnn, ansatz, device, x, y, dataset_size, num_params, num_qubits
        )
        return

    if args["draw"]:
        draw(qcnn, ansatz, device, num_qubits)
        return

    if args["validate"]:
        validate(
            model_data_file,
            ansatz,
            device,
            num_qubits,
            os.path.join(img_folder, "validation.pdf")
        )
        return
