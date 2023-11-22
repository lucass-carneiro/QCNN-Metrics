"""qcnn.

Usage:
  qcnn process <model-id> <dataset-size> [--max-iters=<iters>] [--abstol=<abstol>] [--fisher-samples=<samples>] [--quantum]
  qcnn archive <model-name>
  qcnn (-h | --help)
  qcnn --version

Options:
  --max-iters=<iters>         Maximun number of steps to take while training [default: 1000].
  --abstol=<abstol>           Absolute tolerance. Training stops if abs(cost) < abstol [default: 1.0e-4].
  --fisher-samples=<samples>  How many times to compute Fisher matrices using random parameters [default: 100].
  --quantum                   Plots the spectrum of the Quantum Fisher Matrix, instead of the classical spectrum   
  -h --help                   Show this screen.
  --version                   Show version.
"""

from docopt import docopt

import model_0
import model_1
import model_2
import model_3
import model_4
import model_5

import subprocess
import shutil
import os


def archive(args):
    model_name = args["<model-name>"]
    tar_name = model_name + ".tar"
    gzip_name = tar_name + ".gz"

    subprocess.run([
        "tar",
        "cf",
        tar_name,
        os.path.join("img", model_name),
        os.path.join("data", model_name)
    ])

    subprocess.run([
        "gzip",
        "-9",
        tar_name
    ])

    if not os.path.exists("archive"):
        os.mkdir("archive")

    shutil.move(gzip_name, os.path.join("archive", gzip_name))


def main(args):
    if (args["process"]):
        match int(args["<model-id>"]):
            case 0:
                model_0.process(args)
            case 1:
                model_1.process(args)
            case 2:
                model_2.process(args)
            case 3:
                model_3.process(args)
            case 4:
                model_4.process(args)
            case 5:
                model_5.process(args)

    elif (args["archive"]):
        archive(args)


# Required in order to keep subprocesses from launching recursivelly
if __name__ == '__main__':
    args = docopt(__doc__, version="qcnn 1.0")
    main(args)
