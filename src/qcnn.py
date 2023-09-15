#!python
"""qcnn.

Usage:
  qcnn train <model-id> <dataset-size> [--max-iters=<iters>] [--abstol=<abstol>] [--fisher-samples=<samples>] [--plot-cost]
  qcnn fisher-spectrum <model-id>
  qcnn draw <model-id> <dataset-size>
  qcnn (-h | --help)
  qcnn --version

Options:
  --max-iters=<iters>         Maximun number of steps to take while training [default: 1000].
  --abstol=<abstol>           Absolute tolerance. Training stops if abs(cost) < abstol [default: 1.0e-4].
  --fisher-samples=<samples>  How many times to compute Fisher matrices using random parameters [default: 100].
  --plot-cost                 Plots the cost Vs iteration plot for the training procedure.
  -h --help                   Show this screen.
  --version                   Show version.
"""

from docopt import docopt
import model_0


def main(arguments):
    match int(arguments["<model-id>"]):
        case 0:
            model_0.main(arguments)


# Required in order to keep subprocesses from launching recursivelly
if __name__ == '__main__':
    arguments = docopt(__doc__, version="qcnn 1.0")
    main(arguments)
