from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from unitary_blocks import vatan_williams
from circuit_assembly import new_convolutional_layer

new_convolutional_layer(4, "c1", 3, vatan_williams).draw(output = "mpl", filename = "circuit.pdf")