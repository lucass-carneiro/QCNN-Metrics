"""
This module defines the actual `ansatzes` supported in the code.
All `ansatzes` are specializations of the `ansatz.Ansatz` abstract base class.
"""

import ansatz

from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers

import pennylane as qml


class AnsatzSEL(ansatz.Ansatz):
    """
    This anstaz uses Pauli X rotations as encoding and [strongly entangling layers](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html)
    as trainable blocks
    """

    def __init__(self, num_qubits: int, num_layers: int, random_generator):
        """
        Initialize the object

        Parameters:
          num_qubits (int): The number of qubits to use in the quantum circuit.
          num_layers (int): The number of layers to use.
          random_generator (numpy.random.Generator): A random number generator to use for trainable parameter initialization.
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.param_shape = (2, num_layers, num_qubits, 3)
        self.weights = 2 * np.pi * \
            random_generator.random(size=self.param_shape)

    def S(self, x):
        """
        The enconder block for the ansatz circuit, using Pauli X rotations

        Parameters:
        x (array): An array-like object containing the datapoints to encode.
        """
        for w in range(self.num_qubits):
            qml.RX(x, wires=w)

    def W(self, theta):
        """
        The trainable block for the ansatz circuit, using `StronglyEntanglingLayers`

        Parameters:
          theta (array): An array-like object containing the list of trainable parameters.
        """
        StronglyEntanglingLayers(theta, wires=range(self.num_qubits))

    def ansatz(self, weights, x=None):
        """
        The ansatz circuit containing enconding and trainable blocks.
        The circuit computs the expected value of the Pauli sigma z operator

        Parameters:
          weights (array): An object containing the list of trainable parameters.
          x (array): An object containing the list of datapoints to encode.

        Returns:
          (pennylane.ExpectationMP): The expected value of the quantum circuit
        """
        self.W(weights[0])
        self.S(x)
        self.W(weights[1])
        return qml.expval(qml.PauliZ(wires=0))


class AnsatzConv(ansatz.Ansatz):
    """
    This anstaz uses Pauli X rotations as encoding and convolutional layers from the `conv_layer` module
    as trainable blocks
    """

    def __init__(self, num_qubits: int, conv_layer, random_generator):
        """
        Initialize the object

        Parameters:
          num_qubits (int): The number of qubits to use in the quantum circuit.
          conv_layer (any): The convolutional layer to use in the circuit.
          random_generator (numpy.random.Generator): A random number generator to use for trainable parameter initialization.
        """
        self.num_qubits = num_qubits
        self.conv_layer = conv_layer
        self.param_shape = (2, num_qubits, conv_layer.ppb)
        self.weights = 2 * np.pi * \
            random_generator.random(size=self.param_shape)

    def S(self, x):
        """
        The enconder block for the ansatz circuit, using Pauli X rotations

        Parameters:
        x: 
          An array-like object containing the datapoints to encode.
        """
        for w in range(self.num_qubits):
            qml.RX(x, wires=w)

    def conv_block(self, p):
        """
        Constructs the trainable part of the circuit.
        Each convolutional layer is applied to a pair of quibits.
        As a "boundary condition", the last quibit pairs with the first.
        """
        qml.Barrier(wires=range(self.num_qubits))
        for i in range(self.num_qubits):
            self.conv_layer.layer(p[i], [i, (i + 1) % self.num_qubits])
        qml.Barrier(wires=range(self.num_qubits))

    def W(self, theta):
        """
        The trainable block for the ansatz circuit, using convolutional layers

        Parameters:
          theta (array): An array-like object containing the list of trainable parameters.
        """
        self.conv_block(theta)

    def ansatz(self, weights, x=None):
        """
        The ansatz circuit containing enconding and trainable blocks.
        The circuit computs the expected value of the Pauli sigma z operator

        Parameters:
          weights (array): An object containing the list of trainable parameters.
          x (array): An object containing the list of datapoints to encode.

        Returns:
          (pennylane.ExpectationMP): The expected value of the quantum circuit
        """
        self.W(weights[0])
        self.S(x)
        self.W(weights[1])
        return qml.expval(qml.PauliZ(wires=0))
