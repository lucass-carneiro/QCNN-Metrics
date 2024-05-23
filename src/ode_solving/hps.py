"""Hagen-Poiseuille Quantum Solver

Usage:
  hps.py <config-file>
  hps.py (-h | --help)
  hps.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

import hagen_poiseuille as hp
import model_folders as mf
import optimizer_params as op
import draw_and_plot as plt
import optimize as opt
import ansatzes as ans

from conv_layers import HurKimPark9 as conv_layer

from pennylane import numpy as np
from pennylane.numpy.random import Generator, MT19937

import docopt

import tomllib

def main(args):
    with open(args["<config-file>"], "rb") as f:
        config_file = tomllib.load(f)

    # Configuration
    num_qubits = config_file["computer"]["num_qubits"]
    num_layers = config_file["computer"]["num_layers"]
    ansatz = config_file["computer"]["ansatz"]
    
    dataset_size = config_file["dataset"]["dataset_size"]
    batch_size = config_file["dataset"]["batch_size"]

    optimizer = config_file["training"]["optimizer"]
    max_iters = config_file["training"]["max_iters"]
    abstol = config_file["training"]["abstol"]
    step_size = config_file["training"]["step_size"]

    x0 = config_file["domain"]["x0"]
    xf = config_file["domain"]["xf"]
    
    G = config_file["hagen-poiseuille-params"]["G"]
    R = config_file["hagen-poiseuille-params"]["R"]
    mu = config_file["hagen-poiseuille-params"]["mu"]

    folder_name = f"{config_file["output"]["folder_name"]}_{optimizer}_{ansatz}"

    # These variables are used when using the kokkos backend
    #os.environ["OMP_PROC_BIND"] = "spread"
    #os.environ["OMP_PLACES"] = "threads"
    #os.environ["OMP_NUM_THREADS"] = str(num_qubits)

    # Data folders
    folders = mf.ModelFolders(folder_name)

    # Optimization params
    params = op.OptimizerParams(max_iters, abstol, step_size, batch_size)

    # Random generator
    random_generator = Generator(MT19937(seed=100))

    # Problem to solve
    hp_problem = hp.HagenPoiseuille(x0, xf, G, R, mu)

    # Ansatz
    if ansatz == "sel":
        ansatz_type = ans.AnsatzSEL(num_qubits, num_layers, random_generator)
    elif ansatz == "conv":
        ansatz_type = ans.AnsatzConv(num_qubits, conv_layer, random_generator)
    else:
        print(f"Unknown ansatz \"{ansatz}\"")
        exit(1)

    # Sampling points (global coordinates)
    x = np.linspace(
        hp_problem.map.global_start, 
        hp_problem.map.global_end,
        num=dataset_size,
        endpoint=True
    )

    # Draw circuit
    plt.draw_circuit(
        folders,
        ansatz_type.ansatz,
        num_qubits,
        ansatz_type.weights,
        0.0
    )
    
    # Optimize
    if optimizer == "numpy":
        opt.numpy_optimize(
            folders,
            ansatz_type.ansatz,
            num_qubits,
            ansatz_type.weights,
            x,
            params,
            hp_problem,
            random_generator
        )
    elif optimizer == "torch":
        opt.torch_optimize(
            folders,
            ansatz_type.ansatz,
            num_qubits,
            ansatz_type.weights,
            x,
            params,
            hp_problem,
            random_generator
        )
    else:
        print(f"Unrecognized optimizer \"{optimizer}\"")
        exit(1)

    # Plots
    plt.recover_and_plot(
        folders,
        hp_problem.map,
        ansatz_type.ansatz,
        x,
        num_qubits
    )


if __name__ == "__main__":
    args = docopt.docopt(
        __doc__,
        version="Hagen-Poiseuille Quantum Solver 1.0"
    )
    
    main(args)
