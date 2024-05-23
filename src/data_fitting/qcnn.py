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
  --quantum                   Plots the spectrum of the Quantum Fisher Matrix, instead of the classical spectrum.
  -h --help                   Show this screen.
  --version                   Show version.
"""

from docopt import docopt

import data_fitting_models.model_0 as model_0
import data_fitting_models.model_1 as model_1
import data_fitting_models.model_2 as model_2
import data_fitting_models.model_3 as model_3
import data_fitting_models.model_4 as model_4
import data_fitting_models.model_5 as model_5
import data_fitting_models.model_6 as model_6
import data_fitting_models.model_7 as model_7
import data_fitting_models.model_8 as model_8
import data_fitting_models.model_9 as model_9
import data_fitting_models.model_10 as model_10
import data_fitting_models.model_11 as model_11
import data_fitting_models.model_12 as model_12

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
            case 6:
                model_6.process(args)
            case 7:
                model_7.process(args)
            case 8:
                model_8.process(args)
            case 9:
                model_9.process(args)
            case 10:
                model_10.process(args)
            case 11:
                model_11.process(args)
            case 12:
                model_12.process(args)

    elif (args["archive"]):
        archive(args)


# Required in order to keep subprocesses from launching recursivelly
if __name__ == '__main__':
    args = docopt(__doc__, version="qcnn 1.0")
    main(args)
