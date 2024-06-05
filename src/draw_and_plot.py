import domain_map as dm
import output as out

import matplotlib as mpl
import matplotlib.pyplot as plt

import pennylane as qml

from pennylane import numpy as np

import h5py

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


# def plot_cost(output: out.Output, last_iter, cost_data):
#     fig_name = "cost.pdf"
#     fig_path = os.path.join(output.output_name, fig_name)

#     logger.info(f"Plotting cost data up until iteration {last_iter}")

#     plt.close("all")

#     plt.plot(range(last_iter + 1), cost_data,
#              color=line_color, linewidth=line_thickness)

#     plt.xlabel("Iterations", fontsize=font_size)
#     plt.ylabel("Cost", fontsize=font_size)

#     plt.tight_layout()
#     plt.savefig(fig_path)


# def plot_circuit_function(folders: mf.ModelFolders, map: dm.DomainMap, circuit, weights, data, num_qubits):
#     print("Plotting circuit function")

#     device = qml.device("default.qubit", wires=num_qubits, shots=None)
#     node = qml.QNode(circuit, device)

#     f = [node(weights, x=map.global2local(x_)) for x_ in data]

#     plt.close("all")

#     plt.plot(data, f, color=line_color, linewidth=line_thickness)

#     plt.xlabel("x", fontsize=font_size)
#     plt.ylabel("f(x)", fontsize=font_size)

#     plt.tight_layout()
#     plt.savefig(os.path.join(folders.img_folder, "trained.pdf"))


# def recover_and_plot(folders: mf.ModelFolders, map: dm.DomainMap, ansatz, x, num_qubits):
#     with h5py.File(folders.training_data_file, "r") as f:
#         td = f["trainig_data"]
#         checkpoints = td.attrs["checkpoints"]

#         cost_data = []
#         iter = None
#         w = None

#         for i in range(checkpoints):
#             cpt_group = "checkpoint_{:03d}".format(i)

#             c = list(f.get("trainig_data/{}/cost".format(cpt_group)))
#             cost_data.extend(c)

#             w = np.array(
#                 f.get("trainig_data/{}/weights".format(cpt_group)))

#             iter = f["trainig_data"][cpt_group].attrs["last_iteration"]

#         plot_cost(folders, iter, cost_data)
#         plot_circuit_function(folders, map, ansatz, w, x, num_qubits)
