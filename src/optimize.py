import model_folders as mf
import optimizer_params as op
import hdf5_io as io

import pennylane as qml
import pennylane.numpy as np

import torch

import os

def torch_optimize(folders: mf.ModelFolders, circuit, num_qubits, weights, data, params: op.OptimizerParams, problem, random_generator):
    # Quantum device
    device = qml.device(
        "default.qubit.torch",
        torch_device="cuda",
        wires=num_qubits,
        shots=None
    )

    node = qml.QNode(
        circuit,
        device, 
        interface="torch",
        diff_method="backprop"
    )

    # Recovery
    if os.path.exists(folders.training_data_file):
        print("Recovering previous training data")
        first_iter, weights = io.recover_training_data(folders)
    else:
        first_iter = 0

    last_iter = first_iter + params.max_iters

    # Initial data
    cost_data = []
    stopping_criteria = "max iterations reached"

    N_data = len(data)


    # Pytorch definitions
    weights_torch = torch.tensor(weights, requires_grad=True, device="cuda")

    if params.batch_size == 0:
        batch_data = torch.tensor(data, requires_grad=False, device="cuda")
        N_batch = N_data
    
    opt = torch.optim.LBFGS(
        [weights_torch],
        lr=params.step
    )
    
    # Optimization loop
    for i in range(first_iter, last_iter):
        if params.batch_size != 0:
            batch_indices = random_generator.integers(
                1,
                N_data - 2,
                size=params.batch_size,
                endpoint=True
            )

            batch_data = torch.tensor(data[batch_indices], requires_grad=False, device="cuda")
            N_batch = params.batch_size
        
        def torch_cost():
                opt.zero_grad()
                c = torch.sqrt(problem.cost(node, weights_torch, batch_data, N_batch))
                c.backward()
                return c
        
        # save, and print the current cost
        c = torch_cost().item()
        cost_data.append(c)

        print("Loss in teration", i, "=", c)

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            opt.step(torch_cost)

    # Results
    print("Training results:")
    print("  Stopping criteria: ", stopping_criteria)
    print("  Iterations:", i + 1)
    print("  Final cost value:", cost_data[-1])

    # Save Data
    io.save_training_data(
        folders,
        stopping_criteria,
        first_iter,
        i,
        cost_data,
        weights_torch.cpu().detach().numpy()
    )


def numpy_optimize(folders: mf.ModelFolders, circuit, num_qubits, weights, data, params: op.OptimizerParams, problem, random_generator):
    # Quantum device
    device = qml.device(
        "lightning.kokkos",
        wires=num_qubits,
        shots=None
    )

    node = qml.QNode(
        circuit,
        device, 
        diff_method="adjoint"
    )

    # Recovery
    if os.path.exists(folders.training_data_file):
        print("Recovering previous training data")
        first_iter, weights = io.recover_training_data(folders)
    else:
        first_iter = 0

    last_iter = first_iter + params.max_iters

    # Initial data
    cost_data = []

    opt = qml.AdamOptimizer(params.step)
    #opt = qml.SPSAOptimizer(params.step)
    stopping_criteria = "max iterations reached"

    N_data = len(data)

    for i in range(first_iter, last_iter):
        if params.batch_size != 0:
            batch_indices = random_generator.integers(
                1,
                N_data - 2,
                size=params.batch_size,
                endpoint=True
            )

            batch_data = data[batch_indices]
            N_batch = params.batch_size
        else:
            batch_data = data
            N_batch = N_data

        # save, and print the current cost
        c = np.sqrt(problem.cost(node, weights, batch_data, N_batch))
        cost_data.append(c)

        print("Loss in teration", i, "=", c)

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            weights = opt.step(
                lambda w: np.sqrt(problem.cost(node, w, data, N_data)),
                weights
            )

    # Results
    print("Training results:")
    print("  Stopping criteria: ", stopping_criteria)
    print("  Iterations:", i + 1)
    print("  Final cost value:", cost_data[-1])

    # Save Data
    io.save_training_data(folders, stopping_criteria, first_iter, i, cost_data, weights)