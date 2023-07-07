from unitary_blocks import vatan_williams, qiskit_pooling
from circuit_assembly import new_QCNN_circuit

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit.library import ZFeatureMap

from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN

num_qubits = 4

# Feature map
feature_map = ZFeatureMap(num_qubits)

# QCNN ansatz circuit
ansatz = new_QCNN_circuit(num_qubits, 3, 3, vatan_williams, qiskit_pooling)

# Full Circuit Assembly
qcnn_qc = QuantumCircuit(num_qubits)
qcnn_qc.compose(feature_map, inplace=True)
qcnn_qc.compose(ansatz, inplace=True)
qcnn_qc.draw(output="mpl", filename="circuit.pdf")


# Estimator NN
qcnn = EstimatorQNN(
    circuit=qcnn_qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)


def callback_graph(weights, obj_func_eval):
    # callback function that draws a live plot when the .fit() method is called
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


# Regressor
regressor = NeuralNetworkRegressor(
    neural_network=qcnn,
    loss="squared_error",
    optimizer=L_BFGS_B(maxiter=100),
    callback=callback_graph,
)


# draw(output="mpl", filename="circuit.pdf")
