"""
Model 2:
Type: 3
Block: HurKimPark1
Data: Line with constant slope
"""

from conv_layers import HurKimPark9 as conv_layer
from model_type_2 import ModelType2 as ModelType

import pennylane.numpy as np


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    A = 1.0 / np.sqrt(np.dot(x, x))
    x = A * x

    y = 0.5 * x

    return x, y


def process(args):
    dataset_size = int(args["<dataset-size>"])
    max_iters = int(args["--max-iters"])
    abstol = float(args["--abstol"])
    fisher_samples = int(args["--fisher-samples"])
    quantum = bool(args["--quantum"])

    training_x, training_y = new_dataset(-1.0, 1.0, dataset_size)
    validation_x, validation_y = new_dataset(1.0, 3.0, dataset_size)

    model = ModelType(
        "model_2",
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
