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
Model 1:
Type: 0
Block: HurKimPark3
Data: Line with constant slope
Notes:
 * Good for fitting up to 8 data points
"""

from data_fitting_models.conv_layers import HurKimPark3 as conv_layer
from data_fitting_models.model_type_0 import ModelType0 as ModelType

import pennylane.numpy as np


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    y = 0.5 * x

    A = 1.0 / np.sqrt(np.dot(x, x))
    B = 1.0 / np.sqrt(np.dot(y, y))

    x = A * x
    y = B * y

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
        "model_1",
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
