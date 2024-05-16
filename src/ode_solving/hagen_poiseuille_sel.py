import hagen_poiseuille as hp
import model_folders as mf
import optimizer_params as op
import draw_and_plot as plt
import optimize as opt

import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
from pennylane.numpy.random import Generator, MT19937

import numpy as np

def main():
    # Configuration
    num_qubits = 5
    trainable_block_layers = 3
    dataset_size = 20
    batch_size = 10

    folder_name = "hagen_poiseuille_sel"

    max_iters = 100
    abstol = 1.0e-2
    step_size = 1.0e-2

    x0 = 0.0
    xf = 1.0
    G = 1.0
    R = 1.0
    mu = 1.0

    # Data folders
    folders = mf.ModelFolders(folder_name)

    # Optimization params
    params = op.OptimizerParams(max_iters, abstol, step_size, batch_size)

    # Quantum device
    device = qml.device(
        "default.qubit.torch",
        torch_device="cuda",
        wires=num_qubits,
        shots=None
    )

    # Problem to solve
    hp_problem = hp.HagenPoiseuille(x0, xf, G, R, mu)

    # Sampling points (global coordinates)
    x = np.linspace(
        hp_problem.map.global_start, 
        hp_problem.map.global_end,
        num=dataset_size,
        endpoint=True
    )
    
    # Initial weights
    random_generator = Generator(MT19937(seed=100))
    
    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * random_generator.random(size=param_shape)

    # Ansatz
    def ansatz(weights, x=None):
        StronglyEntanglingLayers(weights[0], wires=range(num_qubits))
    
        for w in range(num_qubits):
            qml.RX(x, wires=w)
    
        StronglyEntanglingLayers(weights[1], wires=range(num_qubits))
    
        return qml.expval(qml.PauliZ(wires=0))

    # Draw circuit
    plt.draw_circuit(folders, ansatz, device, weights, 0.0)
    
    # Optimize
    opt.torch_optimize(folders, ansatz, device, weights, x, params, hp_problem, random_generator)

    # Plots
    plt.recover_and_plot(folders, hp_problem.map, ansatz, device, x)


if __name__ == "__main__":
    main()
