import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
from pennylane import numpy as np
from pennylane.numpy.random import Generator, MT19937


import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os

num_qubits = 5
trainable_block_layers = 3
dataset_size = 20

folder_name = "derivative_test_sel"

max_iters = 10000
abstol = 1.0e-3
step_size = 1.0e-3

font_size = 18
line_thickness = 2.0
line_color = "black"

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size

# These variables are used when using the kokkos backend
os.environ["OMP_PROC_BIND"] = "spread"
os.environ["OMP_PLACES"] = "threads"
os.environ["OMP_NUM_THREADS"] = str(num_qubits)


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


def plot_derivative_error(folders: ModelFolders, circuit, device, weights, data, derivative):
    print("Plotting derivative error")

    f = [circuit_derivative(circuit, device, weights, x_) for x_ in data]

    err = np.abs(derivative - f)

    plt.close("all")

    plt.plot(data, err, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("Derivative Error", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "deriv_error.pdf"))


def plot_second_derivative_error(folders: ModelFolders, circuit, device, weights, data, second_deriv):
    print("Plotting second derivative error")

    f = [circuit_derivative_2(circuit, device, weights, x_) for x_ in data]

    err = np.abs(second_deriv - f)

    plt.close("all")

    plt.plot(data, err, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("Second Derivative Error", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "second_deriv_error.pdf"))


def save_training_data(folders: ModelFolders, stopping_criteria, first_iter, last_iter, cost_data, weights):
    print("Saving training data")

    if not os.path.exists(folders.training_data_file):
        with h5py.File(folders.training_data_file, "w") as f:
            td = f.create_group("trainig_data")
            td.attrs["model_type"] = folders.name
            td.attrs["checkpoints"] = 1

            cpt = td.create_group("checkpoint_000")

            cpt.attrs["stopping_criteria"] = stopping_criteria
            cpt.attrs["first_iteration"] = first_iter
            cpt.attrs["last_iteration"] = last_iter

            cpt.create_dataset("cost", compression="gzip",
                               chunks=True, data=cost_data)
            cpt.create_dataset("weights", compression="gzip",
                               chunks=True, dtype=float, data=weights)
    else:
        with h5py.File(folders.training_data_file, "a") as f:
            td = f["trainig_data"]

            cpt = td.create_group(
                "checkpoint_{:03d}".format(td.attrs["checkpoints"]))

            td.attrs["checkpoints"] += 1

            cpt.attrs["stopping_criteria"] = stopping_criteria
            cpt.attrs["first_iteration"] = first_iter
            cpt.attrs["last_iteration"] = last_iter

            cpt.create_dataset("cost", compression="gzip",
                               chunks=True, data=cost_data)
            cpt.create_dataset("weights", compression="gzip",
                               chunks=True, dtype=float, data=weights)


def recover_training_data(folders: ModelFolders):
    with h5py.File(folders.training_data_file, "r") as f:
        last_checkpoint_group = "checkpoint_{:03d}".format(
            f["trainig_data"].attrs["checkpoints"] - 1)

        i = int(f["trainig_data"]
                [last_checkpoint_group].attrs["last_iteration"]) + 1
        w = np.array(
            f.get("trainig_data/{}/weights".format(last_checkpoint_group)))
        return i, w


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
    # Adjoint differantiation makes the cost function computation
    # substantially faster with lightning backends
    node = qml.QNode(circuit, device, diff_method="adjoint")

    # Recovery
    if os.path.exists(folders.training_data_file):
        print("Recovering previous training data")
        first_iter, weights = recover_training_data(folders)
    else:
        first_iter = 0

    last_iter = first_iter + params.max_iters

    # Initial data
    cost_data = []

    opt = qml.AdamOptimizer(params.step)
    stopping_criteria = "max iterations reached"

    for i in range(first_iter, last_iter):
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
    save_training_data(folders, stopping_criteria,
                       first_iter, i, cost_data, weights)


def circuit_derivative(circuit, device, weights, x):
    node = qml.QNode(circuit, device)
    fp = node(weights, x=(x + np.pi / 2))
    fm = node(weights, x=(x - np.pi / 2))
    return (fp - fm)/2


def circuit_derivative_2(circuit, device, weights, x):
    node = qml.QNode(circuit, device)
    f = node(weights, x=x)
    fp = node(weights, x=(x + np.pi))
    fm = node(weights, x=(x - np.pi))
    return (fm + fp - 2.0 * f) / 4


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


def main():
    # Data folders
    folders = ModelFolders(folder_name)

    # Optimization params
    params = OptimizerParams(max_iters, abstol, step_size)

    # Quantum device
    device = qml.device("lightning.kokkos", wires=num_qubits, shots=None)

    # Sampling points
    x = np.linspace(-1, 1, num=dataset_size, endpoint=True)

    # Solution to the Hagen Poiseuille equation in the [-1, 1] range
    prefactor = 1.0  # G * R^2 / mu
    y = prefactor * (1 - x) * (3 + x) / 16.0
    yp = - prefactor * (1 + x) / 8.0
    ypp = np.full(len(y), -prefactor / 8.0)

    # Initial weights
    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * Generator(MT19937(seed=100)).random(size=param_shape)

    # Fit
    draw_circuit(folders, entangling_circuit, device, weights, 0.0)
    fit_to_target(folders, entangling_circuit, device, weights, x, y, params)

    # Plots
    with h5py.File(folders.training_data_file, "r") as f:
        td = f["trainig_data"]
        checkpoints = td.attrs["checkpoints"]

        cost_data = []
        iter = None
        w = None

        for i in range(checkpoints):
            cpt_group = "checkpoint_{:03d}".format(i)

            c = list(f.get("trainig_data/{}/cost".format(cpt_group)))
            cost_data.extend(c)

            w = np.array(
                f.get("trainig_data/{}/weights".format(cpt_group)))

            iter = f["trainig_data"][cpt_group].attrs["last_iteration"]

        plot_cost(folders, iter, cost_data)
        plot_fit_error(folders, entangling_circuit, device, w, x, y)
        plot_circuit_function(folders, entangling_circuit, device, w, x)
        plot_derivative_error(folders, entangling_circuit, device, w, x, yp)
        plot_second_derivative_error(
            folders, entangling_circuit, device, w, x, ypp)


if __name__ == "__main__":
    main()
