"""
Draws quantum circuits and plot trained functions.
"""

import problem as prob
import output as out
import config as cfg
import ansatzes as ans
import problem as prob

import matplotlib as mpl
import matplotlib.pyplot as plt

import pennylane as qml
import pennylane.numpy as np

import adios2

import os

import logging
logger = logging.getLogger(__name__)

font_size = 18
line_thickness = 2.0
line_color = "black"
target_color = "red"

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


def draw_circuit(output_name: str, circuit, num_qubits, *args):
    """
    Draws a quantum circuit.

    Parameters:
      output_name (str): The folder where the plot will be created.
      circuit (quantum circuit function): The quantum circuit to draw.
      num_qubits (int): The number of qubits in the circuit.
      args (any): Extra arguments for matplotlib.
    """
    fig_name = "model.pdf"
    fig_path = os.path.join(output_name, fig_name)

    if not os.path.exists(fig_path):
        logger.info(f"Drawing circuit image {fig_name}")
        plt.close("all")

        device = qml.device("default.qubit", wires=num_qubits, shots=None)
        node = qml.QNode(circuit, device)

        fig, _ = qml.draw_mpl(node)(*args)
        fig.savefig(fig_path)


def plot_cost(output_name: str, iterations, cost_data):
    """
    Plots a cost function

    Parameters:
      output_name (str): The folder where the plot will be created.
      iterations (array): List of iteration indices in the cost data.
      cost_data (array): Cost values at each iteration index.
    """
    fig_name = "cost.pdf"
    fig_path = os.path.join(output_name, fig_name)

    logger.info(f"Plotting cost data")

    plt.close("all")

    plt.plot(
        iterations,
        cost_data,
        color=line_color,
        linewidth=line_thickness
    )

    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Cost", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(fig_path)


def plot_trained_function(config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, weights, data):
    """
    Plots the function currently represented in a quantum circuit.

    Parameters:
      config (ConfigData): The configuration data used for training.
      ansatz (Ansatz): The ansatz circuit trained.
      problem (Problem): The type of problem that was solved (trained for).
      weights (array): The trained weights of the quantum circuit.
      data (array): The input domain data.
    """
    fig_name = "trained.pdf"
    fig_path = os.path.join(config.output_folder_name, fig_name)

    logger.info(f"Plotting trained function")

    device = qml.device("default.qubit", wires=config.num_qubits, shots=None)
    node = qml.QNode(ansatz.ansatz, device)

    f = [
        node(weights, x=problem.get_domain_map().global2local(x_))
        for x_ in data
    ]

    plt.close("all")

    plt.plot(data, f, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("f(x)", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(fig_path)


def plot_trained_error(config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, weights, data, target):
    """
    Plots the function currently represented in a quantum circuit together with
    a target function that the circuit should have achieved.

    Parameters:
      config (ConfigData): The configuration data used for training.
      ansatz (Ansatz): The ansatz circuit trained.
      problem (Problem): The type of problem that was solved (trained for).
      weights (array): The trained weights of the quantum circuit.
      data (array): The input data.
      target (array): The target data, i.e., the data the circuit should have obtained if training was perfect.
    """
    fig_name = "trained_error.pdf"
    fig_path = os.path.join(config.output_folder_name, fig_name)

    logger.info(f"Plotting trained function")

    device = qml.device("default.qubit", wires=config.num_qubits, shots=None)
    node = qml.QNode(ansatz.ansatz, device)

    f = [
        node(weights, x=problem.get_domain_map().global2local(x_))
        for x_ in data
    ]

    plt.close("all")

    plt.plot(data, f, color=line_color,
             linewidth=line_thickness, label="Trained")
    plt.plot(data, target, color=target_color,
             linewidth=line_thickness, label="Target")

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("f(x)", fontsize=font_size)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_path)


def plot_trained_error_abs(config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, weights, data, target):
    """
    Plots the function currently represented in a quantum circuit and computes
    the absolute difference to a target function that the circuit should have
    achieved.

    Parameters:
      config (ConfigData): The configuration data used for training.
      ansatz (Ansatz): The ansatz circuit trained.
      problem (Problem): The type of problem that was solved (trained for).
      weights (array): The trained weights of the quantum circuit.
      data (array): The input data.
      target (array): The target data, i.e., the data the circuit should have obtained if training was perfect.
    """
    fig_name = "trained_error_abs.pdf"
    fig_path = os.path.join(config.output_folder_name, fig_name)

    logger.info(f"Plotting trained function")

    device = qml.device("default.qubit", wires=config.num_qubits, shots=None)
    node = qml.QNode(ansatz.ansatz, device)

    f = [
        node(weights, x=problem.get_domain_map().global2local(x_))
        for x_ in data
    ]

    plt.close("all")

    plt.plot(data, np.abs(target - f),
             color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("Error", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(fig_path)


def recover_and_plot(output_name: str, config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, data, target):
    """
    Recovers training data from an ADIOS2 file and creates the relevant plots
    from it.

    Parameters:
      config (ConfigData): The configuration data used for training.
      ansatz (Ansatz): The ansatz circuit trained.
      problem (Problem): The type of problem that was solved (trained for).
      data (array): The input data.
      target (array): The target data, i.e., the data the circuit should have obtained if training was perfect.
    """
    # Find all .bp files
    file_list = list(
        filter(
            lambda x: os.path.splitext(x)[1] == ".bp",
            os.listdir(config.output_folder_name)
        )
    )

    if len(file_list) == 0:
        logger.error(
            f"Unable to find bp files to recover from in {
                config.output_folder_name}"
        )
        exit(1)

    # Recover cost
    iterations = []
    cost_data = []

    for file in file_list:
        file_path = os.path.join(output_name, file)

        logger.info(f"Recovering cost data from checkpoint file {file_path}")

        with adios2.FileReader(file_path) as s:
            attrs = s.available_attributes()
            vars = s.available_variables()

            steps = int(vars["weights"]["AvailableStepsCount"])
            first_iter = int(attrs["first_iter"]["Value"])

            # cost in all steps
            costs = s.read("cost", step_selection=[0, steps])

            for i in range(len(costs)):
                iterations.append(first_iter + i)
                cost_data.append(costs[i])

    # Recover the weights of the last iteration of the last checkpoint file
    weights = None

    with adios2.FileReader(file_path) as s:
        vars = s.available_variables()
        steps = int(vars["weights"]["AvailableStepsCount"])

        logger.info(
            f"Recovering weight data from checkpoint file {file_path}"
        )

        weights = np.array(
            s.read("weights", step_selection=[steps - 1, 1]),
            requires_grad=True
        )

    plot_cost(output_name, iterations, cost_data)
    plot_trained_function(output_name, config, ansatz, problem, weights, data)

    if target is not None:
        plot_trained_error(output_name, config, ansatz,
                           problem, weights, data, target)
        plot_trained_error_abs(output_name, config, ansatz,
                               problem, weights, data, target)
