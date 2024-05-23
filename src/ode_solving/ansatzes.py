
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers

import pennylane as qml

class AnsatzSEL:
    def __init__(self, num_qubits: int, num_layers: int, random_generator):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.param_shape = (2, num_layers, num_qubits, 3)
        self.weights = 2 * np.pi * random_generator.random(size=self.param_shape)
    
    def S(self, x):
        for w in range(self.num_qubits):
            qml.RX(x, wires=w)


    def W(self, theta):
        StronglyEntanglingLayers(theta, wires=range(self.num_qubits))


    def ansatz(self, weights, x=None):
        self.W(weights[0])
        self.S(x)
        self.W(weights[1])
        return qml.expval(qml.PauliZ(wires=0))
    
class AnsatzConv:
    def __init__(self, num_qubits: int, conv_layer, random_generator):
        self.num_qubits = num_qubits
        self.conv_layer = conv_layer
        self.param_shape = (2, num_qubits, conv_layer.ppb)
        self.weights = 2 * np.pi * random_generator.random(size=self.param_shape)
    
    def S(self, x):
        for w in range(self.num_qubits):
            qml.RX(x, wires=w)

    def conv_block(self, p):
        qml.Barrier(wires=range(self.num_qubits))
        for i in range(self.num_qubits):
            self.conv_layer.layer(p[i], [i, (i + 1) % self.num_qubits])
        qml.Barrier(wires=range(self.num_qubits))

    def W(self, theta):
        self.conv_block(theta)


    def ansatz(self, weights, x=None):
        self.W(weights[0])
        self.S(x)
        self.W(weights[1])
        return qml.expval(qml.PauliZ(wires=0))