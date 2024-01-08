"""
Model Type 3: This follows the following
  https://arxiv.org/abs/2008.08605
  https://github.com/XanaduAI/expressive_power_of_quantum_models/blob/master/tutorial_expressivity_fourier_series.ipynb

Encoding: User defined. Parallel or serial rotations.

Cost: RMS
"""

import pennylane as qml
import pennylane.numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os


class ModelType3:
    def __init__(
            self,
            name: str,
            x,
            y,
            num_qubits,
            model,
            model_fisher):

        self.name = name

        self.x = x
        self.y = y

        self.num_qubits = num_qubits
        self.model = model
        self.model_fisher = model_fisher

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

        self.fisher_data_file = os.path.join(
            self.data_folder, self.name + "_fisher.hdf5"
        )

        self.training_validation_plot_file = os.path.join(
            self.img_folder, "training_validation.pdf"
        )

        self.validation_plot_file = os.path.join(
            self.img_folder, "validation.pdf"
        )

        self.fisher_plot_file = os.path.join(
            self.img_folder, "fisher_spectrum.pdf"
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

    def compute_fisher(self, x, param_shape, fisher_samples=100):
        print("Computing classical fisher")

        node = qml.QNode(self.model_fisher, self.q_device)
        rng = np.random.default_rng()
        samples = []

        for _ in range(fisher_samples):
            params = 2 * np.pi * np.random.random(size=param_shape)
            samples.append(qml.qinfo.classical_fisher(node)(params, x))

        with h5py.File(self.fisher_data_file, "w") as f:
            fm = f.create_group("fisher_matrix")
            fm.create_dataset("cfm", data=samples)

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

        f, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(box_aspect=1))

        # Fit
        ax1.scatter(x, y, label="Input data", color="black")
        ax1.scatter(x, predictions, label="Trained", color="red")

        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend()

        ax1.set_xlabel("$x$", fontsize=font_size)
        ax1.set_ylabel("$y$", fontsize=font_size)

        # Error
        ax2.scatter(x, error, color="black")

        ax1.set_aspect("equal", adjustable="datalim")

        ax2.set_xlabel("$x$", fontsize=font_size)
        ax2.set_ylabel("Training error", fontsize=font_size)

        f.tight_layout()
        f.savefig(self.training_validation_plot_file, bbox_inches="tight")

    def plot_validation_error(self, x, y, font_size=18):
        with h5py.File(self.training_data_file, "r") as f:
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

        f, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(box_aspect=1))

        # Fit
        ax1.scatter(x, y, label="Input data", color="black")
        ax1.scatter(x, predictions, label="Trained", color="red")

        ax1.set_aspect("equal", adjustable="datalim")
        ax1.legend()

        ax1.set_xlabel("$x$", fontsize=font_size)
        ax1.set_ylabel("$y$", fontsize=font_size)

        # Error
        ax2.scatter(x, error, color="black")

        ax1.set_aspect("equal", adjustable="datalim")

        ax2.set_xlabel("$x$", fontsize=font_size)
        ax2.set_ylabel("Validation error", fontsize=font_size)

        f.tight_layout()
        f.savefig(self.validation_plot_file, bbox_inches="tight")

    def plot_fisher_spectrum(self, font_size=18):
        with h5py.File(self.fisher_data_file, "r") as f:
            fm = np.array(f.get("fisher_matrix/cfm"))

        np.set_printoptions(linewidth=200)

        num_mats = len(fm)
        num_eigenvals = len(fm[0])
        avg_eigenvalues = np.zeros(num_eigenvals)

        for mat in fm:
            eigenvalues = np.abs(np.linalg.eigvals(mat))
            avg_eigenvalues = avg_eigenvalues + eigenvalues

        avg_eigenvalues = avg_eigenvalues / num_mats
        avg_eigenvalues = avg_eigenvalues / np.max(avg_eigenvalues)

        print("Normalized avg. eigenvalues")
        print(avg_eigenvalues)

        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
        mpl.rcParams['xtick.labelsize'] = font_size
        mpl.rcParams['ytick.labelsize'] = font_size

        plt.close("all")

        w = np.ones(num_eigenvals) / num_eigenvals
        plt.hist(
            avg_eigenvalues,
            bins=5,
            range=(0.0, 1.0),
            weights=w
        )

        plt.xlabel("Normalized Avg. Eigenvalues", fontsize=font_size)
        plt.ylabel("Normalized counts", fontsize=font_size)

        plt.tight_layout()
        plt.savefig(self.fisher_plot_file)
