"""
Model 5:
Type: 2
Block: HurKimPark6
Data: Shifted straight line
Notes:
 * Good training fit, bad prediction.
 * Fits a shifted line
"""

from conv_layers import HurKimPark6 as conv_layer
from model_type_2 import ModelType2 as ModelType

import pennylane.numpy as np


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    x = x / np.sqrt(np.dot(x, x))

    y = 0.5 * x + 0.5

    return x, y


def process(args):
    dataset_size = int(args["<dataset-size>"])
    max_iters = int(args["--max-iters"])
    abstol = float(args["--abstol"])
    fisher_samples = int(args["--fisher-samples"])
    quantum = bool(args["--quantum"])

    training_x, training_y = new_dataset(-1, 1, dataset_size)
    validation_x, validation_y = new_dataset(2, 4, dataset_size)

    model = ModelType(
        "model_5",
        training_x,
        training_y,
        conv_layer
    )

    model.draw()
    model.optimize(max_iters, abstol)
    model.compute_fishers(fisher_samples)
    model.plot_fisher_spectrum(quantum)
    model.plot_training_error()
    model.plot_validation_error(validation_x, validation_y)
