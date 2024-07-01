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
