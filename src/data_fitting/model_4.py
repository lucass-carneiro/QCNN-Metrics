# QCNN Toolkit - training Quantum Convolutional Neural Networks for
# fitting functions and solving differential equations.
# Copyright (C) 2023-2024  Lucas Timotheo Sanches, Fabio Pereira dos Santos
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Model 4:
Type: 1
Block: HurKimPark6
Data: Shifted straight line
Notes:
 * Good training fit, bad prediction.
 * Fits a shifted line
"""

from data_fitting_models.conv_layers import HurKimPark6 as conv_layer
from data_fitting_models.model_type_1 import ModelType1 as ModelType

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
        "model_4",
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
