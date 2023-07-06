from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from unitary_blocks import vatan_williams
from circuit_assembly import new_CNN

new_CNN(8, vatan_williams) .draw(output="mpl", filename="circuit.pdf")
