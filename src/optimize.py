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


def torch_optimize(out: out.Output, ansatz: ans.Ansatz, problem: prb.Problem, params: op.OptimizerParams, config: cfg.ConfigData, data_in, random_generator):
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
        weights = torch.tensor(out.weights, requires_grad=True, device="cuda")
    else:
        first_iter = 0
        weights = torch.tensor(
            ansatz.weights, requires_grad=True, device="cuda")

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

    data = torch.tensor(data_in, requires_grad=False, device="cuda")
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

            batch_data = torch.tensor(
                data[batch_indices], requires_grad=False, device="cuda")
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
