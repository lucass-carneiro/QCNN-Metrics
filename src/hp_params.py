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
Contains Hagen Poiseuille problem data handlers.
"""


class HagenPoiseuilleParams:
    """
    A databag for storing the needed parameters of the Hagen-Poiseuille problem.

    Attributes:
      G (float): The Hagen-Poiseuille `G` value.
      R (float): The Hagen-Poiseuille `R` value.
      mu (float): The Hagen-Poiseuille `mu` value.
    """

    def __init__(self, G, R, mu):
        """
        A databag for storing the needed parameters of the Hagen-Poiseuille problem.

        Parameters:
          G (float): The Hagen-Poiseuille `G` value.
          R (float): The Hagen-Poiseuille `R` value.
          mu (float): The Hagen-Poiseuille `mu` value.
        """
        self.G = G
        self.R = R
        self.mu = mu
