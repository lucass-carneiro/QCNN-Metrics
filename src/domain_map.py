"""
Defines domain maps: Maps from a global computational domain to a local domain
ranging from [-1, 1] where the actual training takes place and inverse
transformations from local back to global domain.
"""


class DomainMap:
    """
    Defines a general domain map

    Attributes:
      global_start (float): The left boundary of the global domain.
      global_end (float): The right boundary of the global domain.
    """

    def __init__(self, global_start: float, global_end: float):
        """
        Initialize the object.

        Parameters:
          global_start (float): The left boundary of the global domain.
          global_end (float): The right boundary of the global domain.
        """
        self.global_start = global_start
        self.global_end = global_end

    def local2global(self, local_X):
        """
        Converts a point in local domain to global domain.

        Parameters:
          local_X (float): The local domain point.

        Returns:
          (float): The corresponding global domain point.
        """
        return local_X

    def global2local(self, global_x):
        """
        Converts a point in global domain to local domain.

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The corresponding local domain point.
        """
        return global_x

    def dlocal_dglobal(self, global_x):
        """
        The derivative of local coordinates with respect to global coordinates.
        Obtained by computing the derivative of `global2local`

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The derivative of `global2local` in the passed point.
        """
        return 0.0

    def d2local_dglobal2(self, global_x):
        """
        The second derivative of local coordinates with respect to global coordinates.
        Obtained by computing the second derivative of `global2local`

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The second derivative of `global2local` in the passed point.
        """
        return 0.0


class LinearMap(DomainMap):
    """
    Defines a linear domain map, that is, link global and local domains using
    a linear coordiante transformation.

    Attributes:
      global_start (float): The left boundary of the global domain.
      global_end (float): The right boundary of the global domain.
    """

    def local2global(self, local_X):
        """
        Converts a point in local domain to global domain.

        Parameters:
          local_X (float): The local domain point.

        Returns:
          (float): The corresponding global domain point.
        """
        return (self.global_start + self.global_end + (self.global_end - self.global_start) * local_X) / 2.0

    def global2local(self, global_x):
        """
        Converts a point in global domain to local domain.

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The corresponding local domain point.
        """
        return (self.global_start + self.global_end - 2.0 * global_x) / (self.global_start - self.global_end)

    def dlocal_dglobal(self, global_x):
        """
        The derivative of local coordinates with respect to global coordinates.
        Obtained by computing the derivative of `global2local`

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The derivative of `global2local` in the passed point.
        """
        return 2.0 / (self.global_end - self.global_start)

    def d2local_dglobal2(self, global_x):
        """
        The second derivative of local coordinates with respect to global coordinates.
        Obtained by computing the second derivative of `global2local`

        Parameters:
          global_x (float): The global domain point.

        Returns:
          (float): The second derivative of `global2local` in the passed point.
        """
        return 0.0
