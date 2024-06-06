"""Quantum Solver

Usage:
  hps.py <config-file>
  hps.py (-h | --help)
  hps.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""


import hagen_poiseuille as hp
import function_fitting as ff
import output as out
import optimizer_params as op
import config as cfg
import ansatzes as ans
import conv_layers as cvl
import draw_and_plot as plt
import optimize as opt


from pennylane import numpy as np
from pennylane.numpy.random import Generator, MT19937

import docopt

import sys
import logging
logger = logging.getLogger(__name__)


def main(args):
    # Configuration
    logging.basicConfig(
        format="%(levelname)s [%(asctime)s] [Thread: %(thread)d] [Proc: %(process)d] [%(module)s:%(funcName)s in %(filename)s:%(lineno)d]: %(message)s",
        stream=sys.stdout,
        level=logging.INFO
    )

    config_file_path = args["<config-file>"]
    config = cfg.ConfigData(config_file_path)

    # Data folders
    output = out.Output(config.output_folder_name, config_file_path)

    # Optimization params
    params = op.OptimizerParams(
        config.max_iters,
        config.abstol,
        config.step_size,
        config.batch_size
    )

    # Random generator
    random_generator = Generator(MT19937(seed=100))

    # Ansatz
    if config.ansatz == "sel":
        ansatz_type = ans.AnsatzSEL(
            config.num_qubits,
            config.num_layers,
            random_generator
        )
    elif config.ansatz == "conv":
        ansatz_type = ans.AnsatzConv(
            config.num_qubits,
            eval(f"cvl.{config.conv_layer}"),
            random_generator
        )
    else:
        logger.error(f"Unknown ansatz \"{config.ansatz}\"")
        exit(1)

    # Sampling points (global coordinates)
    x = np.linspace(
        config.x0,
        config.xf,
        num=config.dataset_size,
        endpoint=True
    )

    # Problem to solve
    if config.problem_type == "hagen-poiseuille":
        problem = hp.HagenPoiseuille(
            config.x0,
            config.xf,
            config.hp_params.G,
            config.hp_params.R,
            config.hp_params.mu
        )
    elif config.problem_type == "fit":
        problem = ff.FitToFunction(config.x0, config.xf, config.optimizer)
    else:
        logger.error(f"Unknown problem type \"{config.problem_type}\"")
        exit(1)

    # Draw circuit
    plt.draw_circuit(
        output,
        ansatz_type.ansatz,
        config.num_qubits,
        ansatz_type.weights,
        0.0
    )

    # Optimize
    if config.optimizer == "numpy":
        opt.numpy_optimize(
            output,
            ansatz_type,
            problem,
            params,
            config,
            x,
            random_generator
        )
    elif config.optimizer == "torch":
        # opt.torch_optimize(
        #     output,
        #     ansatz_type.ansatz,
        #     config.num_qubits,
        #     ansatz_type.weights,
        #     x,
        #     params,
        #     problem,
        #     random_generator
        # )
        pass
    else:
        logger.error(f"Unrecognized optimizer \"{config.optimizer}\"")
        exit(1)

    # Plots
    plt.recover_and_plot(output, config, ansatz_type, problem, x)


if __name__ == "__main__":
    args = docopt.docopt(
        __doc__,
        version="Quantum Solver 1.0"
    )

    main(args)
