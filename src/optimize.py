import problem as prb
import output as out
import optimizer_params as op
import ansatzes as ans
import config as cfg

import pennylane as qml
import pennylane.numpy as np

import torch

import os

import logging
logger = logging.getLogger(__name__)

# def torch_optimize(folders: mf.ModelFolders, circuit, num_qubits, weights, data, params: op.OptimizerParams, problem, random_generator):
#     # Quantum device
#     device = qml.device(
#         "default.qubit",
#         wires=num_qubits,
#         shots=None
#     )

#     node = qml.QNode(
#         circuit,
#         device,
#         interface="torch",
#         diff_method="backprop"
#     )

#     # Recovery
#     if os.path.exists(folders.training_data_file):
#         print("Recovering previous training data")
#         first_iter, weights = io.recover_training_data(folders)
#     else:
#         first_iter = 0

#     last_iter = first_iter + params.max_iters

#     # Initial data
#     cost_data = []
#     stopping_criteria = "max iterations reached"

#     N_data = len(data)

#     # Pytorch definitions
#     weights_torch = torch.tensor(weights, requires_grad=True, device="cuda")

#     if params.batch_size == 0:
#         batch_data = torch.tensor(data, requires_grad=False, device="cuda")
#         N_batch = N_data

#     opt = torch.optim.LBFGS(
#         [weights_torch],
#         lr=params.step
#     )

#     # Optimization loop
#     for i in range(first_iter, last_iter):
#         if params.batch_size != 0:
#             batch_indices = random_generator.integers(
#                 1,
#                 N_data - 2,
#                 size=params.batch_size,
#                 endpoint=True
#             )

#             batch_data = torch.tensor(
#                 data[batch_indices], requires_grad=False, device="cuda")
#             N_batch = params.batch_size

#         def torch_cost():
#             opt.zero_grad()
#             c = torch.sqrt(problem.cost(
#                 node, weights_torch, batch_data, N_batch))
#             c.backward()
#             return c

#         # save, and print the current cost
#         c = torch_cost().item()
#         cost_data.append(c)

#         print("Loss in teration", i, "=", c)

#         if np.abs(c) < params.abstol:
#             stopping_criteria = "absolute tolerance reached"
#             break
#         else:
#             opt.step(torch_cost)

#     # Results
#     print("Training results:")
#     print("  Stopping criteria: ", stopping_criteria)
#     print("  Iterations:", i + 1)
#     print("  Final cost value:", cost_data[-1])

#     # Save Data
#     io.save_training_data(
#         folders,
#         stopping_criteria,
#         first_iter,
#         i,
#         cost_data,
#         weights_torch.cpu().detach().numpy()
#     )


def numpy_optimize(out: out.Output, ansatz: ans.Ansatz, problem: prb.Problem, params: op.OptimizerParams, config: cfg.ConfigData, data, random_generator):
    # Quantum device
    device = qml.device(
        "lightning.kokkos",
        wires=config.num_qubits,
        shots=None
    )

    node = qml.QNode(
        ansatz.ansatz,
        device,
        diff_method="adjoint"
    )

    # Recovery
    if out.recovered:
        first_iter = out.first_iter
        weights = np.array(out.weights, requires_grad=True)
    else:
        first_iter = 1
        weights = ansatz.weights

    last_iter = first_iter + params.max_iters

    out.output_stream.write_attribute("first_iter", first_iter)
    out.output_stream.write_attribute("last_iter", last_iter - 1)
    out.output_stream.write_attribute("weights_shape", weights.shape)

    opt = qml.AdamOptimizer(params.step)
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

        logger.info(f"Loss in teration {i} = {c}")

        out.output_stream.begin_step()
        out.output_stream.write("iteration", i)
        out.output_stream.write("cost", c)

        out.output_stream.write(
            "weights",
            weights.numpy(),
            weights.shape,
            [0] * len(weights.shape),
            weights.shape
        )

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            weights = opt.step(
                lambda w: np.sqrt(problem.cost(node, w, data, N_data)),
                weights
            )

        out.output_stream.end_step()

    # Results
    logger.info("Training done")
    logger.info(f"Stopping criteria: {stopping_criteria}")
