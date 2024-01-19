"""
Model 7:
Type: 3
Block: HurKimPark8
Data:
Notes:
"""

from conv_layers import HurKimPark8 as conv_layer

from model_type_3 import ModelType3 as ModelType

import pennylane as qml
import pennylane.numpy as np


# Set higher to fit more modes
num_qubits = 3


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    y = (3 * np.pi + 12 * np.cos(x) - 4 * np.cos(3 * x)) / (6 * np.pi)

    return x, y


def conv_block(p):
    qml.Barrier(wires=range(num_qubits))
    conv_layer.layer(p[0], [0, 1])
    conv_layer.layer(p[1], [1, 2])
    conv_layer.layer(p[2], [2, 0])
    qml.Barrier(wires=range(num_qubits))


def S(x):
    """Data encoding circuit block."""
    for w in range(num_qubits):
        qml.RX(x, wires=w)


def W(theta):
    """Trainable circuit block."""
    conv_block(theta)


def entangling_circuit(weights, x=None):
    W(weights[0])
    S(x)
    W(weights[1])
    return qml.expval(qml.PauliZ(wires=0))


def fisher_circuit(weights, x):
    W(weights[0])
    S(x)
    W(weights[1])
    return qml.probs(wires=range(num_qubits))


def process(args):
    dataset_size = int(args["<dataset-size>"])
    max_iters = int(args["--max-iters"])
    abstol = float(args["--abstol"])
    fisher_samples = int(args["--fisher-samples"])

    t_x, t_y = new_dataset(-2 * np.pi, 2 * np.pi, dataset_size)
    v_x, v_y = new_dataset(3 * np.pi, 7 * np.pi, dataset_size)

    model = ModelType(
        "model_8",
        t_x,
        t_y,
        num_qubits,
        entangling_circuit,
        fisher_circuit
    )

    # Initial parameters
    batch_size = 25

    param_shape = (2, num_qubits, conv_layer.ppb)
    weights = 2 * np.pi * np.random.random(size=param_shape)

    # Processing
    model.draw(weights)
    model.optimize(weights, batch_size, max_iters, abstol)
    model.plot_training_error()
    model.plot_validation_error(v_x, v_y)
    model.compute_fisher(0.0, param_shape, fisher_samples)
    model.plot_fisher_spectrum()
