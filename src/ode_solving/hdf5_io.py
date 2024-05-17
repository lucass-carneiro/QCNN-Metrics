
import model_folders as mf

from pennylane import numpy as np

import h5py

import os

def save_training_data(folders: mf.ModelFolders, stopping_criteria, first_iter, last_iter, cost_data, weights):
    print("Saving training data")

    if not os.path.exists(folders.training_data_file):
        with h5py.File(folders.training_data_file, "w") as f:
            td = f.create_group("trainig_data")
            td.attrs["model_type"] = folders.name
            td.attrs["checkpoints"] = 1

            cpt = td.create_group("checkpoint_000")

            cpt.attrs["stopping_criteria"] = stopping_criteria
            cpt.attrs["first_iteration"] = first_iter
            cpt.attrs["last_iteration"] = last_iter

            cpt.create_dataset("cost", compression="gzip",
                               chunks=True, data=cost_data)
            cpt.create_dataset("weights", compression="gzip",
                               chunks=True, dtype=float, data=weights)
    else:
        with h5py.File(folders.training_data_file, "a") as f:
            td = f["trainig_data"]

            cpt = td.create_group(
                "checkpoint_{:03d}".format(td.attrs["checkpoints"]))

            td.attrs["checkpoints"] += 1

            cpt.attrs["stopping_criteria"] = stopping_criteria
            cpt.attrs["first_iteration"] = first_iter
            cpt.attrs["last_iteration"] = last_iter

            cpt.create_dataset("cost", compression="gzip",
                               chunks=True, data=cost_data)
            cpt.create_dataset("weights", compression="gzip",
                               chunks=True, dtype=float, data=weights)


def recover_training_data(folders: mf.ModelFolders):
    with h5py.File(folders.training_data_file, "r") as f:
        last_checkpoint_group = "checkpoint_{:03d}".format(
            f["trainig_data"].attrs["checkpoints"] - 1)

        i = int(f["trainig_data"]
                [last_checkpoint_group].attrs["last_iteration"]) + 1
        w = np.array(
            f.get("trainig_data/{}/weights".format(last_checkpoint_group)))
        return i, w