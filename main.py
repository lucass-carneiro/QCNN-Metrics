import numpy as np

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from scipy.optimize import minimize

import regression as reg
import unitary_blocks as ubs


def main():

    # create dataset
    num_param = 4
    a = np.random.rand(1)
    x = np.linspace(-np.pi, np.pi, num=num_param)
    y = a * x

    # VQC, ansatz, cost operator and hadamard test of cost operator
    # vqc = new_QCNN_circuit(2, 3, 3, ubs.vatan_williams, ubs.qiskit_pooling)
    vqc = QuantumCircuit(int(np.log2(num_param)))
    ubs.vatan_williams(vqc, 0, 1, ParameterVector("p0", 3))

    vqc.draw(output="mpl", filename="vqc.svg")
    ansatz = reg.make_ansatz(vqc, x)
    cost_operator = reg.make_cost_operator(ansatz, y)
    H_test_cost_op = reg.add_hadamard_test(cost_operator)

    # Simulator
    sim = Aer.get_backend("statevector_simulator")
    sim.set_options(precision="double")

    # Minimizer
    print("Minimizing")
    out = minimize(
        lambda params: reg.global_cost(H_test_cost_op, sim, params),
        x0=[0.5] * H_test_cost_op.num_parameters,
        method="L-BFGS-B",
        options={'maxiter': 1000},
        tol=1e-6
    )
    print(out)

    # Result comparison
    print("---------- Result comparison ----------")
    print("# 1:abs(obtained) 2:abs(expected) 3:error")

    ansatz_eval = np.abs(np.real(reg.eval_pqc(ansatz, sim, out.x)))
    normalized_y = np.abs(y / np.linalg.norm(y))
    reg.save_fit(x, y, ansatz, sim, out.x)

    assert len(ansatz_eval) == len(normalized_y)

    for i in range(len(ansatz_eval)):
        print(
            np.real(ansatz_eval[i]),
            normalized_y[i],
            np.abs(np.real(ansatz_eval[i]) - normalized_y[i]),
            sep="    "
        )

    # Dump cost function
    # print("Dumping cost")
    # reg.dump_cost(8.0, 65, 3, H_test_cost_op, sim)


# Required in order to kepp subprocesses from launching recursivelly
if __name__ == '__main__':
    main()
