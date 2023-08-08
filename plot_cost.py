import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import h5py

font_size = 18

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'Latin Modern Roman'
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size


cost_file = h5py.File("cost_data.hdf5", "r")
cost_data = np.array(cost_file["cost_data"])
cost_file.close()

vars = [
    "a",
    "b",
    "c",
    "cost"
]

cost_df = pd.DataFrame(cost_data, columns=vars)
plot_df = cost_df.loc[cost_df["c"] == -8.0]

plt.tricontourf(plot_df["a"], plot_df["b"], plot_df["cost"], levels=100)

plt.xlabel("$a$", fontsize=font_size)
plt.ylabel("$b$", fontsize=font_size)

plt.colorbar()

plt.tight_layout()
plt.show()
