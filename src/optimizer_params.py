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
Contains optimizer related configurations
"""


class OptimizerParams:
    """
    A databag for holding circuit optimizer related data.

    Attributes:
        max_iters (float): The maximun number of steps the optimizer may take.
        abstol (float): The absolute tolerence of the error function below which training stops.
        step (float): Initial step size for the optimizer algorithim.
        batch_size (int): Size of the subdata array to use for batching. If `0`, batching is disabled.
    """

    def __init__(self, max_iters: int, abstol: float, step: float, batch_size: int):
        """
        Initialize the object.

        Parameters:
            max_iters (float): The maximun number of steps the optimizer may take.
            abstol (float): The absolute tolerence of the error function below which training stops.
            step (float): Initial step size for the optimizer algorithim.
            batch_size (int): Size of the subdata array to use for batching. If `0`, batching is disabled.
        """
        self.max_iters = max_iters
        self.abstol = abstol
        self.step = step
        self.batch_size = batch_size
