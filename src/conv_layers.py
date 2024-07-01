"""
Contains all convolutional layer circuits that can be used for training.
"""

import pennylane as qml
from pennylane import numpy as np

from dataclasses import dataclass


@dataclass
class VatanWilliams:
    """
    Convolutional layer defined in https://arxiv.org/pdf/quant-ph/0308006.pdf

    Attributes:
      name (str): The name of the convolutional layer ("Vatan - Williams").
      ppb (int): The number of trainable parameters per convolutional block (3).
    """

    name: str = "Vatan - Williams"
    ppb: int = 3

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RZ(np.pi / 2, wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[2], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(-np.pi / 2, wires=w[0])


@dataclass
class FreeVatanWilliams:
    """
    Convolutional layer defined in https://arxiv.org/pdf/quant-ph/0308006.pdf.
    Modified so taht all roations are free trainable parameters.

    Attributes:
      name (str): The name of the convolutional layer ("Free Vatan - Williams").
      ppb (int): The number of trainable parameters per convolutional block (5).
    """

    name: str = "Free Vatan - Williams"
    ppb: int = 5

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RZ(p[0], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(p[1], wires=w[0])
        qml.RY(p[2], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[3], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(p[4], wires=w[0])


@dataclass
class HurKimPark1:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (1)").
      ppb (int): The number of trainable parameters per convolutional block (2).
    """

    name: str = "Hur - Kim - Park (1)"
    ppb: int = 2

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])


@dataclass
class HurKimPark2:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (2)").
      ppb (int): The number of trainable parameters per convolutional block (2).
    """

    name: str = "Hur - Kim - Park (2)"
    ppb: int = 2

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.Hadamard(wires=w[0])
        qml.Hadamard(wires=w[1])
        qml.CZ(wires=[w[0], w[1]])
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])


@dataclass
class HurKimPark3:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (3)").
      ppb (int): The number of trainable parameters per convolutional block (4).
    """

    name: str = "Hur - Kim - Park (3)"
    ppb: int = 4

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RY(p[2], wires=w[0])
        qml.RY(p[3], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])


@dataclass
class HurKimPark4:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (4)").
      ppb (int): The number of trainable parameters per convolutional block (6).
    """

    name: str = "Hur - Kim - Park (4)"
    ppb: int = 6

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CRZ(p[2], wires=[w[1], w[0]])
        qml.RY(p[3], wires=w[0])
        qml.RY(p[4], wires=w[1])
        qml.CRZ(p[5], wires=[w[0], w[1]])


@dataclass
class HurKimPark5:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (5)").
      ppb (int): The number of trainable parameters per convolutional block (6).
    """

    name: str = "Hur - Kim - Park (5)"
    ppb: int = 6

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CRX(p[2], wires=[w[1], w[0]])
        qml.RY(p[3], wires=w[0])
        qml.RY(p[4], wires=w[1])
        qml.CRX(p[5], wires=[w[0], w[1]])


@dataclass
class HurKimPark6:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (6)").
      ppb (int): The number of trainable parameters per convolutional block (6).
    """

    name: str = "Hur - Kim - Park (6)"
    ppb: int = 6

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[2], wires=w[0])
        qml.RY(p[3], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[4], wires=w[0])
        qml.RY(p[5], wires=w[1])


@dataclass
class HurKimPark7:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (7)").
      ppb (int): The number of trainable parameters per convolutional block (10).
    """

    name: str = "Hur - Kim - Park (7)"
    ppb: int = 10

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])
        qml.RZ(p[2], wires=w[0])
        qml.RZ(p[3], wires=w[1])
        qml.CRZ(p[4], wires=[w[1], w[0]])
        qml.CRZ(p[5], wires=[w[0], w[1]])
        qml.RX(p[6], wires=w[0])
        qml.RX(p[7], wires=w[1])
        qml.RZ(p[8], wires=w[0])
        qml.RZ(p[9], wires=w[1])


@dataclass
class HurKimPark8:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (8)").
      ppb (int): The number of trainable parameters per convolutional block (10).
    """

    name: str = "Hur - Kim - Park (8)"
    ppb: int = 10

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])
        qml.RZ(p[2], wires=w[0])
        qml.RZ(p[3], wires=w[1])
        qml.CRX(p[4], wires=[w[1], w[0]])
        qml.CRX(p[5], wires=[w[0], w[1]])
        qml.RX(p[6], wires=w[0])
        qml.RX(p[7], wires=w[1])
        qml.RZ(p[8], wires=w[0])
        qml.RZ(p[9], wires=w[1])


@dataclass
class HurKimPark9:
    """
    Convolutional layer defined in https://arxiv.org/abs/2108.00661.

    Attributes:
      name (str): The name of the convolutional layer ("Hur - Kim - Park (9)").
      ppb (int): The number of trainable parameters per convolutional block (15).
    """

    name: str = "Hur - Kim - Park (9)"
    ppb: int = 15

    def layer(p, w):
        """
        Circuit defining the convolutional layer.

        Parameters:
          p (array): List of values for the trainable parameters of the circuit.
          w (array): List of qubit indices where the circuit is to be applied to.
        """
        # U30
        qml.RZ(p[0], wires=w[0])
        qml.RX(-np.pi/2, wires=w[0])
        qml.RZ(p[1], wires=w[0])
        qml.RX(np.pi/2, wires=w[0])
        qml.RZ(p[2], wires=w[0])

        # U31
        qml.RZ(p[3], wires=w[1])
        qml.RX(-np.pi/2, wires=w[1])
        qml.RZ(p[4], wires=w[1])
        qml.RX(np.pi/2, wires=w[1])
        qml.RZ(p[5], wires=w[1])

        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[6], wires=w[0])
        qml.RZ(p[7], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RY(p[8], wires=w[0])
        qml.CNOT(wires=[w[0], w[1]])

        # U30
        qml.RZ(p[9], wires=w[0])
        qml.RX(-np.pi/2, wires=w[0])
        qml.RZ(p[10], wires=w[0])
        qml.RX(np.pi/2, wires=w[0])
        qml.RZ(p[11], wires=w[0])

        # U31
        qml.RZ(p[12], wires=w[1])
        qml.RX(-np.pi/2, wires=w[1])
        qml.RZ(p[13], wires=w[1])
        qml.RX(np.pi/2, wires=w[1])
        qml.RZ(p[14], wires=w[1])
