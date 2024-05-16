import model_folders as mf
import optimizer_params as op
import hdf5_io as io

import numpy as np
import pennylane as qml

import torch

import os

def torch_optimize(folders: mf.ModelFolders, circuit, device, weights, data, params: op.OptimizerParams, problem, random_generator):
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
    
    opt = torch.optim.LBFGS([weights_torch], lr=params.step)

    # Optimization loop
    for i in range(first_iter, last_iter):
        batch_indices = random_generator.integers(
            1,
            N_data - 2,
            size=params.batch_size,
            endpoint=True
        )

        batch_data = torch.tensor(data[batch_indices], requires_grad=False, device="cuda")
        
        # save, and print the current cost
        c = problem.cost(node, weights_torch, batch_data, params.batch_size).item()
        cost_data.append(c)

        print("Loss in teration", i, "=", c)

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            def torch_cost():
                opt.zero_grad()
                c = problem.cost(node, weights_torch, batch_data, params.batch_size)
                c.backward()
                return c
            
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