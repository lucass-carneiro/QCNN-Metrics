import numpy as np

from qiskit import Aer
from qiskit.circuit.library import PauliFeatureMap
from scipy.optimize import minimize

import regression as reg
import unitary_blocks as ubs


def main():

    # create dataset
    num_param = 4
    a = 1.0
    b = 0.0
    x = np.linspace(-1.0, 1.0, num=num_param)
    y = a * x + b

    # VQC, ansatz, cost operator and hadamard test of cost operator
    vqc = ubs.new_vatan_williams()
    ansatz = reg.make_ansatz(vqc, x)
    cost_operator = reg.make_cost_operator(ansatz, y)
    H_test_cost_op = reg.add_hadamard_test(cost_operator)

    # Simulator
    sim = Aer.get_backend('statevector_simulator')

    # Minimizer
    out = minimize(
        lambda params: reg.global_cost(H_test_cost_op, sim, params),
        x0=[-8.0, -3.0, -1.0],
        method="L-BFGS-B",
        options={'maxiter': 1000},
        tol=1e-8
    )
    print(out)

    # Bracketing

    # Result comparison
    print("---------- Result comparison ----------")
    print("# 1:obtained 2:expected 3:error")

    ansatz_eval = reg.eval_pqc(ansatz, sim, out.x)
    normalized_y = y / np.linalg.norm(y)

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
