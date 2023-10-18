import pennylane as qml
from pennylane import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py


def new_device(dataset_size):
    num_qubits = int(np.log2(dataset_size))

    # Shots = None allows for analytic (unnafected by sampling) statistics in this simulator
    q_device = qml.device("default.qubit", wires=num_qubits, shots=None)

    return q_device, num_qubits


def draw(func, device, name, *args):
    def func_with_ret(*args):
        func(*args)
        return qml.state()

    node = qml.QNode(func_with_ret, device)
    fig, _ = qml.draw_mpl(node)(*args)
    fig.savefig(name)


def plot_cost(cost_data, max_iters, name, font_size=18):
    mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size

    plt.close("all")

    plt.plot(range(max_iters), cost_data)

    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Cost", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(name)


def optimize(ansatz, x, H, q_device, num_params, arguments):
    max_iters = int(arguments["--max-iters"])
    abstol = float(arguments["--abstol"])

    # Cost and cost node
    def cost(p):
        ansatz(x, p)
        return qml.expval(H)

    cost_node = qml.QNode(cost, q_device, interface="numpy")

    # Optimization parameters and cost vector
    cost_data = []
    params = np.random.normal(0, np.pi, num_params, requires_grad=True)

    # Optimize
    opt = qml.SPSAOptimizer(maxiter=max_iters)

    print("Optimizing")

    stopping_criteria = "max iterations reached"

    for i in range(max_iters):
        params, loss = opt.step_and_cost(cost_node, params)
        cost_data.append(loss)

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


def classical_fisher(qcnn, num_params, num_qubits, fisher_samples):
    print("Computing classical fisher")

    # Ansatz state and node
    q_device = qml.device("default.qubit", wires=num_qubits, shots=None)

    def ansatz_func(p):
        qcnn(p)
        return qml.probs(wires=range(num_qubits))

    ansatz_node = qml.QNode(ansatz_func, q_device)

    samples = []
    rng = np.random.default_rng()

    for _ in range(fisher_samples):
        params = rng.uniform(low=-1.0, high=1.0, size=num_params)
        samples.append(qml.qinfo.classical_fisher(ansatz_node)(params))

    return samples


def quantum_fisher(qcnn, num_params, num_qubits, fisher_samples):
    print("Computing quantum Fisher")

    # QCNN state and node
    q_device = qml.device("default.qubit", wires=num_qubits + 1, shots=None)

    def qcnn_func(p):
        qcnn(p)
        return qml.probs(wires=range(num_qubits))

    qcnn_node = qml.QNode(qcnn_func, q_device)

    samples = []
    rng = np.random.default_rng()

    for _ in range(fisher_samples):
        params = rng.uniform(low=-1.0, high=1.0, size=num_params)
        samples.append(qml.qinfo.quantum_fisher(qcnn_node)(params))

    return samples


def save_data(name, i, cost_data, params, x, y, cfm, qfm):
    with h5py.File(name, "w") as f:
        td = f.create_group("trainig_data")
        td.create_dataset("iterations", dtype=int, data=range(i + 1))
        td.create_dataset("cost", data=cost_data)
        td.create_dataset("params", dtype=float, data=params)
        td.create_dataset("training_set", dtype=float, data=x)
        td.create_dataset("target_set", dtype=float, data=y)

        fm = f.create_group("fisher_matrix")
        fm.create_dataset("cfm", data=cfm)
        fm.create_dataset("qfm", data=qfm)


def plot_fisher_spectrum(data_file, plot_file, quantum=False, font_size=18):
    with h5py.File(data_file, "r") as f:
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
    # mpl.rcParams['font.family'] = 'Latin Modern Roman'
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
    plt.savefig(plot_file)


def get_dataset_size(data_file):
    with h5py.File(data_file, "r") as f:
        return len(f.get("trainig_data/training_set"))


def validate(data_file, ansatz, q_device, plot_file, font_size=18):
    with h5py.File(data_file, "r") as f:
        x = np.array(f.get("trainig_data/training_set"))
        y = np.array(f.get("trainig_data/target_set"))
        p = np.array(f.get("trainig_data/params"))

    # QCNN state and node
    def ansatz_func(P):
        ansatz(x, P)
        return qml.state()

    ansatz_node = qml.QNode(ansatz_func, q_device)
    x_theta = np.real(ansatz_node(p))
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
