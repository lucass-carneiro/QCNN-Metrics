import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
from pennylane import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os

font_size = 18
line_thickness = 2.0
line_color = "black"

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


class ModelFolders:
    def __init__(self, name: str):
        self.name = name

        self.img_folder = os.path.join("img", self.name)
        self.data_folder = os.path.join("data", self.name)

        self.training_data_file = os.path.join(
            self.data_folder, self.name + "_training.hdf5"
        )

        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)


class OptimizerParams:
    def __init__(self, max_iters: int, abstol: float, step: float):
        self.max_iters = max_iters
        self.abstol = abstol
        self.step = step


def draw_circuit(folders: ModelFolders, circuit, device, *args):
    print("Drawing circuit.")

    plt.close("all")

    node = qml.QNode(circuit, device)

    fig, _ = qml.draw_mpl(node)(*args)
    fig.savefig(os.path.join(folders.img_folder, "model.pdf"))


def plot_cost(folders: ModelFolders, last_iter, cost_data):
    print("Plotting cost data")

    plt.close("all")

    plt.plot(range(last_iter + 1), cost_data,
             color=line_color, linewidth=line_thickness)

    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Cost", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "cost.pdf"))


def plot_circuit_function(folders: ModelFolders, circuit, device, weights, data):
    print("Plotting circuit function")

    node = qml.QNode(circuit, device)

    f = [node(weights, x=x_) for x_ in data]

    plt.close("all")

    plt.plot(data, f, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("f(x)", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "trained.pdf"))


def plot_fit_error(folders: ModelFolders, circuit, device, weights, data, targets):
    print("Plotting training error")

    node = qml.QNode(circuit, device)

    f = [node(weights, x=x_) for x_ in data]

    err = np.abs(targets - f)

    plt.close("all")

    plt.plot(data, err, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("Fit Error", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "error.pdf"))


def save_training_data(folders: ModelFolders, stopping_criteria, last_iter, cost_data, weights):
    print("Saving training data")

    with h5py.File(folders.training_data_file, "w") as f:
        td = f.create_group("trainig_data")

        td.attrs["model_type"] = folders.name
        td.attrs["stopping_criteria"] = stopping_criteria
        td.attrs["iterations"] = last_iter

        td.create_dataset("cost", data=cost_data)
        td.create_dataset("weights", dtype=float, data=weights)


def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss = loss + (t - p) ** 2
    loss = loss / len(targets)
    return np.sqrt(loss)


def cost_to_target(node, weights, data, targets):
    predictions = [node(weights, x=x_) for x_ in data]
    return square_loss(targets, predictions)


def fit_to_target(folders: ModelFolders, circuit, device, weights, data, targets, params: OptimizerParams):
    print("Fitting to target")

    node = qml.QNode(circuit, device)

    # Initial data
    cost_data = []

    opt = qml.AdagradOptimizer(params.step)
    stopping_criteria = "max iterations reached"

    for i in range(params.max_iters):
        # save, and print the current cost
        c = cost_to_target(node, weights, data, targets)
        cost_data.append(c)

        print("Loss in teration", i, "=", c)

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            weights = opt.step(
                lambda w: cost_to_target(node, w, data, targets),
                weights
            )

    # Results
    print("Training results:")
    print("  Stopping criteria: ", stopping_criteria)
    print("  Iterations:", i + 1)
    print("  Final cost value:", cost_data[-1])

    # Save Data
    save_training_data(folders, stopping_criteria, i, cost_data, weights)


def circuit_derivative(circuit, device, weights, x):
    node = qml.QNode(circuit, device)
    fp = node(weights, x=(x + np.pi / 2))
    fm = node(weights, x=(x - np.pi / 2))
    return (fp - fm)/2


def process():
    num_qubits = 3
    dataset_size = 100

    # Data folders
    folders = ModelFolders("derivative_test")

    # Optimization params
    params = OptimizerParams(1000, 1.0e-2, 1.0e-2)

    # Quantum device
    device = qml.device("default.qubit", wires=num_qubits, shots=None)

    # Quantum circuits
    def S(x):
        for w in range(num_qubits):
            qml.RX(x, wires=w)

    def W(theta):
        StronglyEntanglingLayers(theta, wires=range(num_qubits))

    def entangling_circuit(weights, x=None):
        W(weights[0])
        S(x)
        W(weights[1])
        return qml.expval(qml.PauliZ(wires=0))

    # Data
    x = np.linspace(-np.pi, np.pi, num=dataset_size, endpoint=True)
    y = np.sin(x)

    # Initial weights
    trainable_block_layers = 3
    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * np.random.random(size=param_shape)

    # draw_circuit(folders, entangling_circuit, device, weights, 0.0)
    # fit_to_target(folders, entangling_circuit, device, weights, x, y, params)

    with h5py.File(folders.training_data_file, "r") as f:
        # iters = f["trainig_data"].attrs["iterations"]
        # cost = np.array(f.get("trainig_data/cost"))
        w = np.array(f.get("trainig_data/weights"))
        # plot_cost(folders, iters, cost)
        # plot_fit_error(folders, entangling_circuit, device, w, x, y)
        # plot_circuit_function(folders, entangling_circuit, device, w, x)
        f = [circuit_derivative(entangling_circuit, device, w, x_) for x_ in x]
        plt.plot(x, f)
        plt.plot(x, np.cos(x))
        plt.show()


process()
