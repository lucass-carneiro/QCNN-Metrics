"""Quantum Solver

Usage:
  hps.py <config-file>
  hps.py (-h | --help)
  hps.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

# These variables are used when using the kokkos backend
import tomllib
import docopt
from pennylane.numpy.random import Generator, MT19937
from pennylane import numpy as np
from conv_layers import HurKimPark9 as conv_layer
import ansatzes as ans
import optimize as opt
import draw_and_plot as plt
import optimizer_params as op
import model_folders as mf
import function_fitting as ff
import hagen_poiseuille as hp


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

    problem_type = config_file["problem"]["problem_type"]

    if problem_type == "hagen-poiseuille":
        G = config_file["hagen-poiseuille-params"]["G"]
        R = config_file["hagen-poiseuille-params"]["R"]
        mu = config_file["hagen-poiseuille-params"]["mu"]

    folder_name = f"{config_file["output"]["folder_name"]}_{optimizer}_{ansatz}"

    # Data folders
    folders = mf.ModelFolders(folder_name)

    # Optimization params
    params = op.OptimizerParams(max_iters, abstol, step_size, batch_size)

    # Random generator
    random_generator = Generator(MT19937(seed=100))

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
        x0,
        xf,
        num=dataset_size,
        endpoint=True
    )

    # Problem to solve
    if problem_type == "hagen-poiseuille":
        problem = hp.HagenPoiseuille(x0, xf, G, R, mu)
    elif problem_type == "fit":
        def f(x):
            return np.exp(x * np.cos(3.0 * np.pi * x)) / 2.0

        problem = ff.FitToFunction(x0, xf, f)
    else:
        print(f"Unknown problem type \"{problem_type}\"")
        exit(1)

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
            problem,
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
            problem,
            random_generator
        )
    else:
        print(f"Unrecognized optimizer \"{optimizer}\"")
        exit(1)

    # Plots
    plt.recover_and_plot(
        folders,
        problem.map,
        ansatz_type.ansatz,
        x,
        num_qubits
    )


if __name__ == "__main__":
    args = docopt.docopt(
        __doc__,
        version="Quantum Solver 1.0"
    )

    main(args)
