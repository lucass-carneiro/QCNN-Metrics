import pennylane as qml
from pennylane import numpy as np

from dataclasses import dataclass


@dataclass
class VatanWilliams:
    """See https://arxiv.org/pdf/quant-ph/0308006.pdf"""

    name: str = "Vatan - Williams"
    ppb: int = 3
    nqubits: int = 2

    def layer(p, w):
        qml.RZ(np.pi / 2, wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])
        qml.RY(p[2], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RZ(-np.pi / 2, wires=w[0])


@dataclass
class HurKimPark1:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (1)"
    ppb: int = 2
    nqubits: int = 2

    def layer(p, w):
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])


@dataclass
class HurKimPark2:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (2)"
    ppb: int = 2
    nqubits: int = 2

    def layer(p, w):
        qml.Hadamard(wires=w[0])
        qml.Hadamard(wires=w[1])
        qml.CZ(wires=[w[0], w[1]])
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])


@dataclass
class HurKimPark3:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (3)"
    ppb: int = 4
    nqubits: int = 2

    def layer(p, w):
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CNOT(wires=[w[1], w[0]])
        qml.RY(p[2], wires=w[0])
        qml.RY(p[3], wires=w[1])
        qml.CNOT(wires=[w[0], w[1]])


@dataclass
class HurKimPark4:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (4)"
    ppb: int = 4

    def layer(p, w):
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CRZ(p[2], wires=[w[1], w[0]])
        qml.RY(p[3], wires=w[0])
        qml.RY(p[4], wires=w[1])
        qml.CRZ(p[5], wires=[w[0], w[1]])


@dataclass
class HurKimPark5:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (5)"
    ppb: int = 6

    def layer(p, w):
        qml.RY(p[0], wires=w[0])
        qml.RY(p[1], wires=w[1])
        qml.CRX(p[2], wires=[w[1], w[0]])
        qml.RY(p[3], wires=w[0])
        qml.RY(p[4], wires=w[1])
        qml.CRX(p[5], wires=[w[0], w[1]])


@dataclass
class HurKimPark6:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (6)"
    ppb: int = 6

    def layer(p, w):
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
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (7)"
    ppb: int = 10

    def layer(p, w):
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])
        qml.RZ(p[2], wires=w[0])
        qml.RZ(p[3], wires=w[1])
        qml.CRZ(p[4], wires=[w[1], w[0]])
        qml.CRZ(p[5], wires=[w[0], w[0]])
        qml.RX(p[6], wires=w[0])
        qml.RX(p[7], wires=w[1])
        qml.RZ(p[8], wires=w[0])
        qml.RZ(p[9], wires=w[1])


@dataclass
class HurKimPark8:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (8)"
    ppb: int = 10

    def layer(p, w):
        qml.RX(p[0], wires=w[0])
        qml.RX(p[1], wires=w[1])
        qml.RZ(p[2], wires=w[0])
        qml.RZ(p[3], wires=w[1])
        qml.CRX(p[4], wires=[w[1], w[0]])
        qml.CRX(p[5], wires=[w[0], w[0]])
        qml.RX(p[6], wires=w[0])
        qml.RX(p[7], wires=w[1])
        qml.RZ(p[8], wires=w[0])
        qml.RZ(p[9], wires=w[1])


@dataclass
class HurKimPark9:
    """See https://arxiv.org/abs/2108.00661"""

    name: str = "Hur - Kim - Park (9)"
    ppb: int = 15

    def layer(p, w):
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
