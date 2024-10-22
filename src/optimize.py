"""
Contains the main optimizer functionality for training circuits
"""

import problem as prb
import output as out
import optimizer_params as op
import ansatz as ans
import config as cfg

import pennylane as qml
import pennylane.numpy as np

import torch

import logging
logger = logging.getLogger(__name__)


def torch_optimize(out: out.Output, ansatz: ans.Ansatz, problem: prb.Problem, params: op.OptimizerParams, config: cfg.ConfigData, data_in, random_generator):
    """
    Optimizes the quantum circuit to solve a problem using the Torch library.

    Parameters:
        out (Output): The output object controlling where the training data will be saved to / recovered from.
        ansatz (Ansatz): The trainable ansatz circuit for solving the problem.
        problem (Problem): The problem object containing the problem type to be solved.
        params (OptimizerParams): The configuration parameters of the optimizer.
        config (ConfigData): The program configuration data.
        data_in (array): The domain data points.
        random_generator (numpy.random.Generator): A random number generator used for initializing the circuit weights.
    """
    # Quantum device
    device = qml.device(
        "default.qubit",
        wires=config.num_qubits,
        shots=None
    )

    node = qml.QNode(
        ansatz.ansatz,
        device,
        interface="torch",
        diff_method="backprop"
    )

    # Recovery
    if out.recovered:
        first_iter = out.first_iter
        if config.use_cuda:
            weights = torch.tensor(
                out.weights, requires_grad=True, device="cuda")
        else:
            weights = torch.tensor(
                out.weights, requires_grad=True, device="cpu")
    else:
        first_iter = 0
        if config.use_cuda:
            weights = torch.tensor(
                ansatz.weights, requires_grad=True, device="cuda")
        else:
            weights = torch.tensor(
                ansatz.weights, requires_grad=True, device="cpu")

    last_iter = first_iter + params.max_iters

    out.output_stream.write_attribute("first_iter", first_iter)
    out.output_stream.write_attribute("last_iter", last_iter - 1)
    out.output_stream.write_attribute("weights_shape", weights.shape)

    opt = torch.optim.LBFGS([weights], lr=params.step)

    # Adaptive LR. See
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        "min",
        0.1,
        5,
        1.0e-4,
        "rel",
        0,
        0.0,
        1.0e-8
    )

    stopping_criteria = "max iterations reached"

    if config.use_cuda:
        data = torch.tensor(data_in, requires_grad=False, device="cuda")
    else:
        data = torch.tensor(data_in, requires_grad=False, device="cpu")

    N_data = len(data)

    # Optimization loop
    for i in range(first_iter, last_iter):
        if params.batch_size != 0:
            batch_indices = random_generator.integers(
                1,
                N_data - 2,
                size=params.batch_size,
                endpoint=True
            )

            if config.use_cuda:
                batch_data = torch.tensor(
                    data[batch_indices], requires_grad=False, device="cuda")
            else:
                batch_data = torch.tensor(
                    data[batch_indices], requires_grad=False, device="cpu")

            N_batch = params.batch_size
        else:
            batch_data = data
            N_batch = N_data

        def torch_cost():
            opt.zero_grad()
            c = torch.sqrt(problem.cost(
                node, weights, batch_data, N_batch))
            c.backward()
            return c

        # save, and print the current cost
        c = torch_cost().item()

        out.output_stream.begin_step()
        out.output_stream.write("iteration", i)
        out.output_stream.write("cost", c)

        out.output_stream.write(
            "weights",
            weights.cpu().detach().numpy(),
            weights.shape,
            [0] * len(weights.shape),
            weights.shape
        )

        out.output_stream.end_step()

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            opt.step(torch_cost)
            sch.step(c)

        logger.info(f"(Loss, LR) in iteration {i} = ({c}, {sch.get_last_lr()})")

    # Results
    logger.info("Training done")
    logger.info(f"Stopping criteria: {stopping_criteria}")


def numpy_optimize(out: out.Output, ansatz: ans.Ansatz, problem: prb.Problem, params: op.OptimizerParams, config: cfg.ConfigData, data, random_generator):
    """
    Optimizes the quantum circuit to solve a problem using the Numpy library.

    Parameters:
        out (Output): The output object controlling where the training data will be saved to / recovered from.
        ansatz (Ansatz): The trainable ansatz circuit for solving the problem.
        problem (Problem): The problem object containing the problem type to be solved.
        params (OptimizerParams): The configuration parameters of the optimizer.
        config (ConfigData): The program configuration data.
        data (array): The domain data points.
        random_generator (numpy.random.Generator): A random number generator used for initializing the circuit weights.
    """
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
        first_iter = 0
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

        out.output_stream.end_step()

        if np.abs(c) < params.abstol:
            stopping_criteria = "absolute tolerance reached"
            break
        else:
            weights = opt.step(
                lambda w: np.sqrt(problem.cost(node, w, data, N_data)),
                weights
            )

        logger.info(f"Loss in teration {i} = {c}")

    # Results
    logger.info("Training done")
    logger.info(f"Stopping criteria: {stopping_criteria}")
