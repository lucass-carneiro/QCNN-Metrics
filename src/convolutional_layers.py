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


def hur_kim_park_4(p, w):
    qml.RY(p[0], wires=w[0])
    qml.RY(p[1], wires=w[1])
    qml.CRZ(p[2], wires=[w[1], w[0]])
    qml.RY(p[3], wires=w[0])
    qml.RY(p[4], wires=w[1])
    qml.CRZ(p[5], wires=[w[0], w[1]])


hur_kim_park_4_ppb = 4


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


def hur_kim_park_7(p, w):
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


hur_kim_park_7_ppb = 10


def hur_kim_park_8(p, w):
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


hur_kim_park_8_ppb = 10


def hur_kim_park_9(p, w):
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


hur_kim_park_8_ppb = 15
