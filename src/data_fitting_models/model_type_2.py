"""
Model Type 2:
Encoding: Amplitude
Cost: RMS
"""

import pennylane as qml
import pennylane.numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py

import os


class ModelType2:
    def __init__(self, name, x, y, conv_layer):
        self.name = name
        self.x = x
        self.y = y
        self.conv_layer = conv_layer

        # Dataset validation
        if len(self.x) != len(self.y):
            raise RuntimeError("x and y data are not of the same size")
        self.dataset_size = len(self.x)

        if not ((self.dataset_size & (self.dataset_size - 1) == 0) and self.dataset_size != 0):
            raise RuntimeError("The dataset size is not a pwoer of 2")

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
        self.num_qubits = int(np.log2(self.dataset_size))

        self.q_device = qml.device(
            "default.qubit",
            wires=self.num_qubits,
            shots=None
        )

        # We assume that number of blocks = number of qubits
        self.num_params = self.conv_layer.ppb * self.num_qubits

    def qcnn(self, p):
        qubit_range = range(self.num_qubits + 1)
        for previous, current in zip(qubit_range, qubit_range[1:]):
            self.conv_layer.layer(
                p,
                [previous % self.num_qubits, current % self.num_qubits]
            )

    def ansatz(self, x, p):
        qml.AmplitudeEmbedding(
            features=x,
            wires=range(self.num_qubits)
        )
        self.qcnn(p)
        return qml.state()

    def draw(self):
        def qcnn_func():
            self.qcnn([0.0] * self.conv_layer.ppb * self.num_qubits)
            return qml.state()

        def ansatz_func():
            return self.ansatz(
                self.x,
                [0.0] * self.conv_layer.ppb * self.num_qubits
            )

        node = qml.QNode(qcnn_func, self.q_device)
        fig, _ = qml.draw_mpl(node)()
        fig.savefig(os.path.join(self.img_folder, "qcnn.pdf"))

        node = qml.QNode(ansatz_func, self.q_device)
        fig, _ = qml.draw_mpl(node)()
        fig.savefig(os.path.join(self.img_folder, "ansatz.pdf"))

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

    def save_training_data(self, i, cost_data, params):
        with h5py.File(self.training_data_file, "w") as f:
            td = f.create_group("trainig_data")

            td.attrs["model_type"] = 2
            td.attrs["convolutional_block"] = self.conv_layer.name

            td.create_dataset("iterations", dtype=int, data=range(i + 1))
            td.create_dataset("cost", data=cost_data)
            td.create_dataset("params", dtype=float, data=params)
            td.create_dataset("training_set", dtype=float, data=self.x)
            td.create_dataset("target_set", dtype=float, data=self.y)

    def save_fisher(self, cfm, qfm):
        with h5py.File(self.fisher_data_file, "w") as f:
            fm = f.create_group("fisher_matrix")
            fm.create_dataset("cfm", data=cfm)

            if qfm != None:
                fm.create_dataset("qfm", data=qfm)

    def optimize(self, max_iters: int, abstol: float):
        cost_node = qml.QNode(self.ansatz, self.q_device, interface="numpy")
        target = self.y.astype(complex)

        def cost(p):
            trained = cost_node(self.x, p)

            cost_diff = target - trained
            cost_diffS = np.conjugate(cost_diff)
            yi_m_xi_2 = np.real(cost_diff * cost_diffS)

            return np.sqrt(np.sum(yi_m_xi_2) / self.num_params)

        # Optimization parameters and cost vector
        cost_data = []

        rng = np.random.default_rng()
        params = rng.uniform(low=-np.pi, high=np.pi, size=self.num_params)

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

        # Plot cost
        self.plot_cost(
            os.path.join(self.img_folder, "cost.pdf"),
            cost_data,
            i + 1
        )

        # Save Data
        self.save_training_data(i, cost_data, params)

    def classical_fisher(self, fisher_samples):
        print("Computing classical fisher")

        def func(p):
            self.qcnn(p)
            return qml.probs(wires=range(self.num_qubits))

        ansatz_node = qml.QNode(func, self.q_device)

        samples = []
        rng = np.random.default_rng()

        for _ in range(fisher_samples):
            params = rng.uniform(low=-np.pi, high=np.pi, size=self.num_params)
            samples.append(qml.qinfo.classical_fisher(ansatz_node)(params))

        return samples

    def quantum_fisher(self, fisher_samples):
        print("Computing quantum Fisher")

        # QCNN state and node
        q_device = qml.device(
            "default.qubit",
            wires=self.num_qubits + 1,
            shots=None
        )

        def qcnn_func(p):
            self.qcnn(p)
            return qml.probs(wires=range(self.num_qubits))

        qcnn_node = qml.QNode(qcnn_func, q_device)

        samples = []
        rng = np.random.default_rng()

        for _ in range(fisher_samples):
            params = rng.uniform(low=-np.pi, high=np.pi, size=self.num_params)
            samples.append(qml.qinfo.quantum_fisher(qcnn_node)(params))

        return samples

    def compute_fishers(self, skip_fisher=True, fisher_samples=100):
        cfm = self.classical_fisher(fisher_samples)

        if not skip_fisher:
            qfm = self.quantum_fisher(fisher_samples)
        else:
            qfm = None

        self.save_fisher(cfm, qfm)

    def plot_training_error(self, font_size=18):
        with h5py.File(self.training_data_file, "r") as f:
            x = np.array(f.get("trainig_data/training_set"))
            y = np.array(f.get("trainig_data/target_set"))
            p = np.array(f.get("trainig_data/params"))

        ansatz_node = qml.QNode(self.ansatz, self.q_device)
        x_theta = np.real(ansatz_node(x, p))
        print("Original data")
        print(y)
        print("Trained data")
        print(x_theta)
        print("Error")
        error = np.abs(np.abs(y) - np.abs(x_theta))
        print(error)

        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
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
        ax2.set_ylabel("Training error", fontsize=font_size)

        f.tight_layout()
        f.savefig(self.training_validation_plot_file, bbox_inches="tight")

    def plot_validation_error(self, x, y, font_size=18):
        with h5py.File(self.training_data_file, "r") as f:
            p = np.array(f.get("trainig_data/params"))

        ansatz_node = qml.QNode(self.ansatz, self.q_device)
        x_theta = np.real(ansatz_node(x, p))
        print("Original data")
        print(y)
        print("Trained data")
        print(x_theta)
        print("Error")
        error = np.abs(np.abs(y) - np.abs(x_theta))
        print(error)

        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'Latin Modern Roman'
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
        ax2.set_ylabel("Validation error", fontsize=font_size)

        f.tight_layout()
        f.savefig(self.validation_plot_file, bbox_inches="tight")

    def plot_fisher_spectrum(self, quantum=False, font_size=18):
        with h5py.File(self.fisher_data_file, "r") as f:
            if quantum:
                fm = np.array(f.get("fisher_matrix/qfm"))
            else:
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
