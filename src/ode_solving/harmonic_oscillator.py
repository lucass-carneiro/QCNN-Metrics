import pennylane as qml
import pennylane.numpy as np
from pennylane.templates import StronglyEntanglingLayers

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os


class ODESolver:
    def __init__(
            self,
            name: str,
            x,
            y,
            num_qubits,
            model):

        self.name = name

        self.x = x
        self.y = y

        self.num_qubits = num_qubits
        self.model = model

        # Dataset validation
        if len(self.x) != len(self.y):
            raise RuntimeError("x and y data are not of the same size")
        self.dataset_size = len(self.x)

        # Data folders
        self.img_folder = os.path.join("img", self.name)
        self.data_folder = os.path.join("data", self.name)

        self.training_data_file = os.path.join(
            self.data_folder, self.name + "_training.hdf5"
        )

        self.training_validation_plot_file = os.path.join(
            self.img_folder, "training_validation.pdf"
        )

        self.training_validation_error_plot_file = os.path.join(
            self.img_folder, "training_validation_error.pdf"
        )

        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # Device creation
        self.q_device = qml.device(
            "default.qubit",
            wires=self.num_qubits,
            shots=None
        )

    def draw(self, params):
        plt.close("all")

        node = qml.QNode(self.model, self.q_device)

        fig, _ = qml.draw_mpl(node)(params, 0.0)
        fig.savefig(os.path.join(self.img_folder, "model.pdf"))

    def plot_cost(self, name, cost_data, max_iters, font_size=18):
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['xtick.labelsize'] = font_size
        mpl.rcParams['ytick.labelsize'] = font_size

        plt.close("all")

        plt.plot(range(max_iters), cost_data)

        plt.xlabel("Iterations", fontsize=font_size)
        plt.ylabel("Cost", fontsize=font_size)

        plt.tight_layout()
        plt.savefig(name)

    def plot_initial(self, name, initial_data, font_size=18):
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['xtick.labelsize'] = font_size
        mpl.rcParams['ytick.labelsize'] = font_size

        plt.close("all")

        plt.plot(self.x, initial_data)
        plt.ylim(-1, 1)

        plt.xlabel(r"$x$", fontsize=font_size)
        plt.ylabel(r"$f_\theta(x)$", fontsize=font_size)

        plt.tight_layout()
        plt.savefig(name)

    def save_training_data(self, stopping_criteria, i, cost_data, params):
        with h5py.File(self.training_data_file, "w") as f:
            td = f.create_group("trainig_data")

            td.attrs["model_type"] = 3
            td.attrs["stopping_criteria"] = stopping_criteria

            td.create_dataset("iterations", dtype=int, data=range(i + 1))
            td.create_dataset("cost", data=cost_data)
            td.create_dataset("params", dtype=float, data=params)
            td.create_dataset("training_set", dtype=float, data=self.x)
            td.create_dataset("target_set", dtype=float, data=self.y)

    def optimize(self, weights, batch_size: int, max_iters: int, abstol: float):
        model_node = qml.QNode(self.model, self.q_device)

        # Cost functions
        def square_loss(targets, predictions):
            loss = 0
            for t, p in zip(targets, predictions):
                loss = loss + (t - p) ** 2
            loss = loss / len(targets)
            return np.sqrt(loss)

        def cost(w, x, y):
            predictions = [model_node(w, x=x_) for x_ in x]
            return square_loss(y, predictions)

        # Initial data
        initial_data = [model_node(weights, xi) for xi in self.x]
        self.plot_initial(
            os.path.join(self.img_folder, "initial.pdf"),
            initial_data
        )

        cost_data = [cost(weights, self.x, self.y)]

        # Optimize
        print("Optimizing")

        # opt = qml.SPSAOptimizer(maxiter=max_iters)
        opt = qml.AdamOptimizer(1.0e-2)
        stopping_criteria = "max iterations reached"

        for i in range(max_iters):
            # select batch of data
            batch_index = np.random.randint(0, len(self.x), (batch_size,))
            x_batch = self.x[batch_index]
            y_batch = self.y[batch_index]

            # update the weights by one optimizer step
            weights = opt.step(lambda w: cost(w, x_batch, y_batch), weights)

            # save, and print the current cost
            c = cost(weights, self.x, self.y)
            cost_data.append(c)
            print("Loss in teration", i, "=", c)

            # Terminate
            if np.abs(c) < abstol:
                stopping_criteria = "absolute tolerance reached"
                break

        # Results
        print("Training results:")
        print("  Stopping criteria: ", stopping_criteria)
        print("  Iterations:", i + 1)
        print("  Final cost value:", cost_data[-1])
        print("  Final training parameters:", weights)

        # Plot cost
        self.plot_cost(
            os.path.join(self.img_folder, "cost.pdf"),
            cost_data,
            i + 2
        )

        # Save Data
        self.save_training_data(stopping_criteria, i, cost_data, weights)

    def plot_training_error(self, font_size=18):
        with h5py.File(self.training_data_file, "r") as f:
            x = np.array(f.get("trainig_data/training_set"))
            y = np.array(f.get("trainig_data/target_set"))
            w = np.array(f.get("trainig_data/params"))

        model_node = qml.QNode(self.model, self.q_device)
        predictions = [model_node(w, x=x_) for x_ in x]

        print("Original data")
        print(y)
        print("Trained data")
        print(predictions)
        print("Error")
        error = np.abs(y - predictions)
        print(error)

        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['xtick.labelsize'] = font_size
        mpl.rcParams['ytick.labelsize'] = font_size

        plt.close("all")

        # Fit
        plt.plot(x, y, label="Input data", color="black")
        plt.plot(x, predictions, label="Trained", color="red")

        plt.legend()

        plt.xlabel("$x$", fontsize=font_size)
        plt.ylabel("$y$", fontsize=font_size)

        plt.tight_layout()
        plt.savefig(self.training_validation_plot_file, bbox_inches="tight")

        plt.close("all")

        # Error
        plt.plot(x, error, color="black")

        plt.xlabel("$x$", fontsize=font_size)
        plt.ylabel("Training error", fontsize=font_size)

        plt.tight_layout()
        plt.savefig(self.training_validation_error_plot_file,
                    bbox_inches="tight")

        plt.close("all")


num_qubits = 4


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size, endpoint=False)
    y = x / np.pi

    return x, y


def S(x):
    """Data encoding circuit block."""
    for w in range(num_qubits):
        qml.RX(x, wires=w)


def W(theta):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(num_qubits))


def entangling_circuit(weights, x=None):
    W(weights[0])
    S(x)
    W(weights[1])
    return qml.expval(qml.PauliZ(wires=0))


def process():
    dataset_size = 100
    max_iters = 100
    abstol = 1.0e-3

    # Data
    x, y = new_dataset(-np.pi, np.pi, dataset_size)
    # x, y = new_dataset(1, dataset_size)

    solver = ODESolver(
        "ode_solver",
        x,
        y,
        num_qubits,
        entangling_circuit
    )

    # Initial parameters
    trainable_block_layers = 5
    batch_size = 25

    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * np.random.random(size=param_shape)

    # Train and save
    solver.draw(weights)
    solver.optimize(weights, batch_size, max_iters, abstol)
    solver.plot_training_error()

    # with h5py.File(solver.training_data_file, "r") as f:
    #     w = np.array(f.get("trainig_data/params"))

    # model_node = qml.QNode(solver.model, solver.q_device)
    # f_x = [model_node(w, x=x_) for x_ in x]
    # f_2_x = [i*i for i in f_x]

    # k = np.fft.rfftfreq(dataset_size, d=1/dataset_size)
    # fp_x = np.fft.irfft(1j * k * np.fft.rfft(f_x), n=dataset_size)

    # lhs = fp_x - f_2_x - f_x

    # plt.plot(x, lhs)
    # plt.show()


process()
