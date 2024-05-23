import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import h5py


def plot_fit(font_size=18):
    plt.close("all")

    mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['font.family'] = 'Latin Modern Roman'
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"
    mpl.rcParams['text.usetex'] = True

    fit_file = h5py.File("data/fit_data.hdf5", "r")
    input_data = np.array(fit_file["input_data"])
    target_data = np.array(fit_file["normalized_target_data"])
    fitted_data = np.array(fit_file["fitted_data"])
    fit_file.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw=dict(box_aspect=1))

    # Data plot
    axes[0].scatter(input_data, fitted_data,
                    label=r"$\left. U(\bm{\theta}) \middle| \bm{x} \right\rangle$"
                    )

    axes[0].scatter(input_data, target_data,
                    label=r"$\left| \bm{y} \right\rangle$    ")

    axes[0].set_xlabel("$x$", fontsize=font_size)
    axes[0].set_ylabel("$y$", fontsize=font_size)

    axes[0].legend()

    # Error plot

    axes[1].scatter(input_data, np.abs(target_data - fitted_data),
                    label=r"$\left. U(\bm{\theta}) \middle| \bm{x} \right\rangle$"
                    )

    axes[1].set_xlabel("$x$", fontsize=font_size)
    axes[1].set_ylabel("Error", fontsize=font_size)

    # Save

    fig.tight_layout()
    fig.savefig("img/fit.svg", bbox_inches="tight")


def plot_cost(font_size=18):
    plt.close("all")

    cost_file = h5py.File("data/cost_data.hdf5", "r")
    cost_data = np.array(cost_file["cost_data"])
    cost_file.close()

    vars = [
        "a",
        "b",
        "c",
        "cost"
    ]

    cval = 1.5
    cost_df = pd.DataFrame(cost_data, columns=vars)
    plot_df = cost_df.loc[cost_df["c"] == cval]

    cnt = plt.tricontourf(
        plot_df["a"], plot_df["b"], plot_df["cost"], levels=np.linspace(0.0, 1.0, 100))

    for c in cnt.collections:
        c.set_edgecolor("face")

    plt.title("Cost function with $c = " + str(cval) + "$", fontsize=font_size)
    plt.xlabel("$a$", fontsize=font_size)
    plt.ylabel("$b$", fontsize=font_size)

    plt.colorbar()

    plt.tight_layout()
    plt.savefig("img/cost.png", dpi=500)
