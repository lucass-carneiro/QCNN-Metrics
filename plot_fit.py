import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import h5py

font_size = 18

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'Latin Modern Roman'
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"
mpl.rcParams['text.usetex'] = True


fit_file = h5py.File("fit_data.hdf5", "r")
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
fig.savefig("fit.svg", bbox_inches="tight")
