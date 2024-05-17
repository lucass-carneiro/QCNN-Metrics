import hagen_poiseuille as hp
import model_folders as mf
import optimizer_params as op
import draw_and_plot as plt
import optimize as opt
import ansatzes as ans

from conv_layers import HurKimPark9 as conv_layer

from pennylane import numpy as np
from pennylane.numpy.random import Generator, MT19937

def main():
    # Configuration
    num_qubits = 5
    trainable_block_layers = 1
    dataset_size = 20
    batch_size = 10

    folder_name = "hagen_poiseuille_sel"

    max_iters = 50
    abstol = 1.0e-2
    step_size = 1.0e-4

    x0 = 0.0
    xf = 1.0
    G = 1.0
    R = 1.0
    mu = 1.0

    # Data folders
    folders = mf.ModelFolders(folder_name)

    # Optimization params
    params = op.OptimizerParams(max_iters, abstol, step_size, batch_size)

    # Random generator
    random_generator = Generator(MT19937(seed=100))

    # Problem to solve
    hp_problem = hp.HagenPoiseuille(x0, xf, G, R, mu)

    # Ansatz
    ansatz_type = ans.AnsatzConv(num_qubits, conv_layer, random_generator)

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

    # Plots
    plt.recover_and_plot(
        folders,
        hp_problem.map,
        ansatz_type.ansatz,
        x,
        num_qubits
    )


if __name__ == "__main__":
    main()
