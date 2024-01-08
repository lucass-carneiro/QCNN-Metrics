"""
Model 6:
Type: 3
Block: HurKimPark6
Data:
Notes:
"""

from conv_layers import VatanWilliams as conv_layer
from pooling_layers import QiskitPooling as pool_layer

from model_type_3 import ModelType3 as ModelType

import pennylane as qml
import pennylane.numpy as np
from pennylane.templates import StronglyEntanglingLayers


def new_dataset(x_min, x_max, dataset_size):
    x = np.linspace(x_min, x_max, num=dataset_size)
    y = np.sin(x)

    return x, y


def qcnn_circuit(p, x):
    # W(p[0]) convolution
    conv_layer.layer(p[0], [0, 1])
    qml.Barrier(wires=[0, 1, 2, 3])

    conv_layer.layer(p[0], [1, 2])
    qml.Barrier(wires=[0, 1, 2, 3])

    conv_layer.layer(p[0], [2, 3])
    qml.Barrier(wires=[0, 1, 2, 3])

    conv_layer.layer(p[0], [3, 0])
    qml.Barrier(wires=[0, 1, 2, 3])

    # S(x)
    qml.RX(x, wires=0)
    qml.RX(x, wires=1)
    qml.RX(x, wires=2)
    qml.RX(x, wires=3)
    qml.Barrier(wires=[0, 1, 2, 3])

    # W(p[1]) pooling
    pool_layer.layer(p[1], [3, 1])
    qml.Barrier(wires=[0, 1, 2, 3])

    pool_layer.layer(p[1], [2, 0])
    qml.Barrier(wires=[0, 1, 2, 3])

    # W(p[2]) convolution
    conv_layer.layer(p[2], [0, 1])
    qml.Barrier(wires=[0, 1, 2, 3])

    conv_layer.layer(p[2], [1, 0])
    qml.Barrier(wires=[0, 1, 2, 3])

    # S(x)
    qml.RX(x, wires=0)
    qml.RX(x, wires=1)
    qml.RX(x, wires=2)
    qml.RX(x, wires=3)
    qml.Barrier(wires=[0, 1, 2, 3])

    # W(p[3]) pooling
    pool_layer.layer(p[3], [1, 0])
    qml.Barrier(wires=[0, 1, 2, 3])

    return qml.expval(qml.PauliZ(wires=0))


num_qubits = 3


def S(x):
    """Data encoding circuit block."""
    for w in range(num_qubits):
        qml.RX(x, wires=w)


def W(theta):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(num_qubits))


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
        "model_6",
        t_x,
        t_y,
        num_qubits,
        entangling_circuit,
        fisher_circuit
    )

    # Initial parameters
    trainable_block_layers = 3
    batch_size = 25

    param_shape = (2, trainable_block_layers, num_qubits, 3)
    weights = 2 * np.pi * np.random.random(size=param_shape)

    # Processing
    model.draw(weights)
    model.optimize(weights, batch_size, max_iters, abstol)
    model.plot_training_error()
    model.plot_validation_error(v_x, v_y)
    model.compute_fisher(0.0, param_shape, fisher_samples)
    model.plot_fisher_spectrum()
