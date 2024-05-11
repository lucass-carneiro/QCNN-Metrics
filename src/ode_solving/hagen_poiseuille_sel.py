import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
from pennylane import numpy as np
from pennylane.numpy.random import Generator, MT19937

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os

num_qubits = 5
trainable_block_layers = 2
dataset_size = 20
batch_size = 10

folder_name = "hagen_poiseuille_sel"

# HP equation parameters
G = 1.0
R = 1.0
mu = 1.0

max_iters = 10000
abstol = 1.0e-6
step_size = 1.0e-4

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

# Coordinate transformations
global_a = 0
global_b = 1.0


def global2local(global_x):
    return (global_a + global_b - 2.0 * global_x) / (global_a - global_b)


def local2global(local_X):
    return (global_a + global_b + (global_b - global_a) * local_X) / 2.0


def dlocal_dglobal(_):
    # Obtained by computing the derivative of global2local
    return 2.0 / (global_b - global_a)


def d2local_dglobal2(_):
    # Obtained by computing the second derivative of global2local
    return 0


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

    f = [node(weights, x=global2local(x_)) for x_ in data]

    plt.close("all")

    plt.plot(data, f, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("f(x)", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "trained.pdf"))


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


def df(node, weights, x):
    fp_2 = node(weights, x=(x + np.pi / 2.0))
    fm_2 = node(weights, x=(x - np.pi / 2.0))
    return (fp_2 + fm_2) / 2.0


def d2f(node, weights, x):
    f = node(weights, x=x)
    fp = node(weights, x=(x + np.pi))
    fm = node(weights, x=(x - np.pi))
    return (fm + fp - 2.0 * f) / 4.0


def cost_int_pointwise(node, weights, x):
    # Get local X
    X = global2local(x)

    # Compute derivatives in local space
    l_dfdX = df(node, weights, x=X)
    l_d2fdX2 = d2f(node, weights, x=X)

    # Compute jacobians
    dldg = dlocal_dglobal(x)
    d2ldg2 = d2local_dglobal2(x)

    # Compute derivatives in global space
    g_dfdx = dldg * l_dfdX
    g_d2fdx2 = dldg * dldg * l_d2fdX2 + d2ldg2 * l_dfdX

    # ODE in global space
    return (g_d2fdx2 + G/mu) * x + g_dfdx


def cost(node, weights, data, N):
    # BCs
    bc_l = (node(weights, x=global2local(global_a)) - G * R**2 / (4.0 * mu))**2
    bc_r = (node(weights, x=global2local(global_b)))**2
    bc_d = (df(node, weights, x=global2local(global_a)))**2

    # Interior cost
    int_cost = sum(cost_int_pointwise(node, weights, x) ** 2 for x in data)

    return np.sqrt((bc_l + bc_r + bc_d + int_cost) / N)


def optimize(folders: ModelFolders, circuit, device, weights, data, params: OptimizerParams):
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

    N_data = len(data)

    for i in range(first_iter, last_iter):
        batch_indices = Generator(MT19937(seed=100)).integers(
            1,
            N_data - 2,
            size=batch_size,
            endpoint=True
        )
        batch_data = data[batch_indices]

        # save, and print the current cost
        c = cost(node, weights, batch_data, batch_size)
        cost_data.append(c)

        print("Loss in teration", i, "=", c)

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            weights = opt.step(
                lambda w: cost(node, w, batch_data, batch_size),
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

    # Sampling points (global coordinates)
    x = np.linspace(global_a, global_b, num=dataset_size, endpoint=True)

    # Initial weights
    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * Generator(MT19937(seed=100)).random(size=param_shape)

    # Solve
    draw_circuit(folders, entangling_circuit, device, weights, 0.0)
    optimize(folders, entangling_circuit, device, weights, x, params)

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
        plot_circuit_function(folders, entangling_circuit, device, w, x)


if __name__ == "__main__":
    main()
