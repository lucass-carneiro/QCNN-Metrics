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

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


def draw_circuit(output: out.Output, circuit, num_qubits, *args):
    fig_name = "model.pdf"
    fig_path = os.path.join(output.output_name, fig_name)

    if not os.path.exists(fig_path):
        logger.info(f"Drawing circuit image {fig_name}")
        plt.close("all")

        device = qml.device("default.qubit", wires=num_qubits, shots=None)
        node = qml.QNode(circuit, device)

        fig, _ = qml.draw_mpl(node)(*args)
        fig.savefig(fig_path)


def plot_cost(output: out.Output, iterations, cost_data):
    fig_name = "cost.pdf"
    fig_path = os.path.join(output.output_name, fig_name)

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


def plot_trained_function(output: out.Output, config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, weights, data):
    fig_name = "trained.pdf"
    fig_path = os.path.join(output.output_name, fig_name)

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


def recover_and_plot(output: out.Output, config: cfg.ConfigData, ansatz: ans.Ansatz, problem: prob.Problem, data):
    # Find all .bp files
    file_list = list(
        filter(
            lambda x: os.path.splitext(x)[1] == ".bp",
            os.listdir(output.output_name)
        )
    )

    if len(file_list) == 0:
        logger.error(
            f"Unable to find bp files to recover from in {output.output_name}"
        )
        exit(1)

    # Recover cost
    iterations = []
    cost_data = []

    for file in file_list:
        file_path = os.path.join(output.output_name, file)

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

    plot_cost(output, iterations, cost_data)
    plot_trained_function(output, config, ansatz, problem, weights, data)
