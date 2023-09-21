import pennylane as qml
from pennylane import numpy as np


# The generalized 2-qubit unitary defined in https://arxiv.org/abs/quant-ph/0308006, Fig 6.
def vatan_williams(p, w):
    qml.RZ(np.pi / 2, wires=w[1])
    qml.CNOT(wires=[w[1], w[0]])
    qml.RZ(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CNOT(wires=[w[0], w[1]])
    qml.RY(p[2], wires=w[1])
    qml.CNOT(wires=[w[1], w[0]])
    qml.RZ(-np.pi / 2, wires=w[0])


vatan_williams_ppb = 3

# The following functions are define based on https://arxiv.org/abs/2108.00661
# Fig. 2 (a) - (i)


def hur_kim_park_1(p, w):
    qml.RY(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CNOT(wires=[w[0], w[1]])


hur_kim_park_1_ppb = 2


def hur_kim_park_2(p, w):
    qml.Hadamard(wires=w[0])
    qml.Hadamard(wires=w[1])
    qml.CZ(wires=[w[0], w[1]])
    qml.RX(p[0], wires=w[0])
    qml.RX(p[1], wires=w[1])


hur_kim_park_2_ppb = 2


def hur_kim_park_3(p, w):
    qml.RY(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CNOT(wires=[w[1], w[0]])
    qml.RY(p[2], wires=w[0])
    qml.RY(p[3], wires=w[1])
    qml.CNOT(wires=[w[0], w[1]])


hur_kim_park_3_ppb = 4


def hur_kim_park_5(p, w):
    qml.RY(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CRX(p[2], wires=[w[1], w[0]])
    qml.RY(p[3], wires=w[0])
    qml.RY(p[4], wires=w[1])
    qml.CRX(p[5], wires=[w[0], w[1]])


hur_kim_park_5_ppb = 6


def hur_kim_park_6(p, w):
    qml.RY(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CNOT(wires=[w[0], w[1]])
    qml.RY(p[2], wires=w[0])
    qml.RY(p[3], wires=w[1])
    qml.CNOT(wires=[w[0], w[1]])
    qml.RY(p[4], wires=w[0])
    qml.RY(p[5], wires=w[1])


hur_kim_park_6_ppb = 6
