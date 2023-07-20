import numpy as np
from data_encondings import amplitude_encode

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit import Aer, execute

from scipy.optimize import minimize


def eval_ansatz(ansatz, sim, params):
    # https://quantumcomputing.stackexchange.com/a/7131
    ansatz_with_params = ansatz.assign_parameters(params)

    job = execute(ansatz_with_params, sim, shots=2000)
    sv = np.array(job.result().get_statevector(ansatz_with_params))
    print(sv)
    return sv


def compute_cost(A_y, alpha_y, ansatz, sim, params):
    sv = eval_ansatz(ansatz, sim, params)

    psi_y = 0.0
    for i, _ in enumerate(alpha_y):
        psi_y += alpha_y[i] * sv[i]
    psi_y *= A_y

    return 1.0 - np.sqrt(np.real(psi_y * np.conjugate(psi_y)))


# create dataset
num_param = 4
a = 1.0
b = 1.0
x = np.linspace(-1.0, 1.0, num=num_param)
y = a * x + b

# Create and init ansatz quantum circuit
A_x, alpha_x, num_qubits = amplitude_encode(x)
A_y, alpha_y, _ = amplitude_encode(y)

ansatz = QuantumCircuit(num_qubits)
ansatz.initialize(A_x * alpha_x)
ansatz.compose(ZZFeatureMap(num_qubits, reps=1), inplace=True)
ansatz.draw(output="mpl", filename="circuit.pdf")

sim = Aer.get_backend('statevector_simulator')

eval_ansatz(ansatz, sim, [1.0, 1.0])


def f(params):
    return compute_cost(A_y, alpha_y, ansatz, sim, params)


def g(x):
    return x[1] - x[0] * x[0] + 1


# out = minimize(g, x0=[1.0, 1.0], method="BFGS", options={'maxiter': 200}, tol=1e-3)

# print(out)

# with open("data.ascii", "w") as file:
#    for a in np.arange(-10.0, 10.0, 0.1):
#        for b in np.arange(-10.0, 10.0, 0.1):
#            print(a, b, f([a, b]), sep="    ", file=file)
