import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
from pennylane import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os


def dx_2(f, h):
    N = len(f)
    fp = np.zeros(N)

    i = 0
    fp[i] = (-3*f[i] + 4*f[i+1] - f[i+2])/(2.0 * h)

    i = N - 1
    fp[i] = (1*f[i-2]-4*f[i-1]+3*f[i])/(2.0 * h)

    for i in range(1, N - 1):
        fp[i] = (f[i + 1] - f[i - 1])/(2.0*h)

    return fp


def dxx_2(f, h):
    N = len(f)
    fpp = [0 for i in range(N)]

    i = 0
    fpp[i] = (f[i]-2*f[i+1]+f[i+2])/(h**2)

    i = N - 1
    fpp[i] = (f[i-2]-2*f[i-1]+f[i])/(h**2)

    for i in range(1, N - 1):
        fpp[i] = (f[i-1]-2*f[i]+f[i+1])/(h**2)

    return fpp


def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss = loss + (t - p) ** 2
    loss = loss / len(targets)
    return np.sqrt(loss)


class ODESolver:
    def __init__(
            self,
            name: str,
            x,
            num_qubits,
            model,
            font_size=18):

        self.name = name

        self.x = x
        self.dx = float(x[1] - x[0])

        self.num_qubits = num_qubits
        self.model = model

        # Dataset validation
        self.dataset_size = len(self.x)

        # Data folders
        self.img_folder = os.path.join("img", self.name)
        self.data_folder = os.path.join("data", self.name)

        self.training_data_file = os.path.join(
            self.data_folder, self.name + "_training.hdf5"
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

        self.font_size = font_size
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['xtick.labelsize'] = self.font_size
        mpl.rcParams['ytick.labelsize'] = self.font_size

    def draw(self, params):
        plt.close("all")

        node = qml.QNode(self.model, self.q_device)

        fig, _ = qml.draw_mpl(node)(params, 0.0)
        fig.savefig(os.path.join(self.img_folder, "model.pdf"))

    def plot_cost(self, name, cost_data, max_iters):
        plt.close("all")

        plt.plot(range(max_iters), cost_data)

        plt.xlabel("Iterations", fontsize=self.font_size)
        plt.ylabel("Cost", fontsize=self.font_size)

        plt.tight_layout()
        plt.savefig(name)

    def plot_model(self, name, initial_data):
        plt.close("all")

        plt.plot(self.x, initial_data)
        plt.ylim(-1, 1)

        plt.xlabel(r"$x$", fontsize=self.font_size)
        plt.ylabel(r"$f_\theta(x)$", fontsize=self.font_size)

        plt.tight_layout()
        plt.savefig(name)

    def save_training_data(self, stopping_criteria, i, cost_data, params, f):
        with h5py.File(self.training_data_file, "w") as f:
            td = f.create_group("trainig_data")

            td.attrs["model_type"] = "ODESolver"
            td.attrs["stopping_criteria"] = stopping_criteria

            td.create_dataset("iterations", dtype=int, data=range(i + 1))
            td.create_dataset("cost", data=cost_data)
            td.create_dataset("params", dtype=float, data=params)
            td.create_dataset("x", dtype=float, data=self.x)
            td.create_dataset("f", dtype=float, data=self.x)

    def optimize(self, weights, max_iters: int, abstol: float):
        model_node = qml.QNode(self.model, self.q_device)

        # Cost function
        # Solves
        # x''(t) + pi/2 x(t) = 0
        # x(-pi) = 0
        # x(pi) = 1
        # Solution: csc(sqrt(2) * pi^(3/2))*Sin(sqrt(pi/2)(pi+t))
        def cost(w):
            # Function and 2nd derivative
            f = [model_node(w, x=x_) for x_ in self.x]
            fpp = dxx_2(f, self.dx)

            # Boundary loss
            bl = square_loss([0, 1], [f[0], f[-1]])

            # Residual loss
            rhs = [fi * (np.pi / 2) for fi in f]
            rl = square_loss(fpp, rhs)

            # Total loss
            return bl + rl

        # Initial data
        f = [model_node(weights, xi) for xi in self.x]
        self.plot_model(os.path.join(self.img_folder, "initial.pdf"), f)

        cost_data = [cost(weights)]

        # Optimize
        print("Optimizing")

        # opt = qml.SPSAOptimizer(maxiter=max_iters)
        opt = qml.AdagradOptimizer(1.0e-3)
        stopping_criteria = "max iterations reached"

        for i in range(max_iters):
            # update the weights by one optimizer step
            weights = opt.step(cost, weights)

            # save, and print the current cost
            c = cost(weights)
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

        # Plot Trained data
        f = [model_node(weights, xi) for xi in self.x]
        self.plot_model(os.path.join(self.img_folder, "final.pdf"), f)

        # Save Data
        self.save_training_data(stopping_criteria, i, cost_data, weights, f)


num_qubits = 4


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
    max_iters = 1000
    abstol = 1.0e-3

    # Data
    # x = np.linspace(-np.pi, np.pi, num=dataset_size, endpoint=False)
    x = np.linspace(-np.pi, np.pi, num=dataset_size, endpoint=True)

    solver = ODESolver(
        "ode_solver",
        x,
        num_qubits,
        entangling_circuit
    )

    # Initial parameters
    trainable_block_layers = 5

    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * np.random.random(size=param_shape)

    # Train and save
    solver.draw(weights)
    solver.optimize(weights, max_iters, abstol)

    # model_node = qml.QNode(solver.model, solver.q_device)
    # f = np.array([model_node(w, x=x_) for x_ in x])

    # # k = np.fft.rfftfreq(dataset_size, d=1/dataset_size)
    # # fp_x = np.fft.irfft(1j * k * np.fft.rfft(f_x), n=dataset_size)

    # # lhs = fp_x - f_2_x - f_x

    # plt.plot(x, dxx_2(np.sin(x), x[1] - x[0]))
    # plt.plot(x, -np.sin(x))
    # plt.show()


process()
