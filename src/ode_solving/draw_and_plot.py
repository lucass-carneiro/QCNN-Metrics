import model_folders as mf 
import domain_map as dm

import matplotlib as mpl
import matplotlib.pyplot as plt

import pennylane as qml

import numpy as np

import h5py

import os

font_size = 18
line_thickness = 2.0
line_color = "black"

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Latin Modern Roman"
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size

def draw_circuit(folders: mf.ModelFolders, circuit, device, *args):
    print("Drawing circuit.")

    plt.close("all")

    node = qml.QNode(circuit, device)

    fig, _ = qml.draw_mpl(node)(*args)
    fig.savefig(os.path.join(folders.img_folder, "model.pdf"))


def plot_cost(folders: mf.ModelFolders, last_iter, cost_data):
    print("Plotting cost data")

    plt.close("all")

    plt.plot(range(last_iter + 1), cost_data,
             color=line_color, linewidth=line_thickness)

    plt.xlabel("Iterations", fontsize=font_size)
    plt.ylabel("Cost", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "cost.pdf"))


def plot_circuit_function(folders: mf.ModelFolders, map: dm.DomainMap, circuit, device, weights, data):
    print("Plotting circuit function")

    node = qml.QNode(circuit, device)

    f = [node(weights, x=map.global2local(x_)) for x_ in data]

    plt.close("all")

    plt.plot(data, f, color=line_color, linewidth=line_thickness)

    plt.xlabel("x", fontsize=font_size)
    plt.ylabel("f(x)", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(os.path.join(folders.img_folder, "trained.pdf"))

def recover_and_plot(folders: mf.ModelFolders, map: dm.DomainMap, ansatz, device, x):
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
        plot_circuit_function(folders, map, ansatz, device, w, x)