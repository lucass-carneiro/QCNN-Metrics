# Intro

This is a python package for training quantum convolutional neural networks. It uses Pennylane for building quantum circuits and can be trained with Numpy or Torch backends. The code also supports checkpoints, recovery and streaming IO via the ADIOS2 library.

# Installing dependencies

The recommended way to install the required dependencies is by creating a virtual environment and using pip. All the package's dependencies are specified in the `requirements.txt` file within the repository. To create a suitable virtual environment, issue the following commands from within the `QCNN-Metrics` repository folder:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Issue python commands using the libraries installed in the virtual environment

deactivate # When done with using the environment
```

Note that the virtual environment does not necessarily need to be named `venv` or located inside the `QCNN-Metrics` folder. Just be sure to point to the correct location of the `requirements.txt` file when creating the environment and to activate the correct environment prior to interacting with the code.

# Using

Activate the virtual environment with the dependencies installed and call the `qcnn.py` script. Issue the `deactivate` command when done. 

To activate the environment, issue

```bash
source venv/bin/activate
```

Where `venv` is the virtual environment folder where dependencies where installed. Next, to see the available options issue

```bash
pyhton src/qcnn.py --help
```

When all operations are done exit out of the virtual environment with

``` bash
deactivate
```

# Parameter files

The operations of `qcnn.py` are controlled via TOML parameter files. Examples of such files can be found in the `parfiles` folder of `QCNN-Metrics`. Bellow is a description of each tunable parameter

| Key | Description | Possible Values |
|:---:|:-----------:|:---------------:|
|`computer/num_qubits`|The number of quibits to use in the circuit| Positive integers
|`circuit/ansatz`| The type of convolutional ansatz to use| `"conv"` for convolutional circuits or `"sel"` for strongly entangling layers.
|`circuit/num_layers`|The number of convolutional layers. Only used if `circuit/ansatz = "sel"`| Positive integers
|`circuit/conv_layer`|The type of convolutional layer to use. Only used if `circuit/ansatz = "conv"`|See `src/conv_layers.py` for possible convolutional layers
|`dataset/dataset_size`|The number of points in the training dataset.|Positive integers
|`dataset/batch_size`|The size of the training batch. If zero, batching is disabled.| Integers
|`training/optimizer`|The provider of the optimization algorithms.|`"numpy"` for Numpy or `"torch"` for Torch
|`training/max_iters`|The maximmun number of training steps to take before exiting.| Integers
|`training/abstol`| This parameter governs the value of the loss function below which the training stops.|Positive reals
|`training/stepsize`| The initial optimization step size| Positive real.
|`domain/x0`|The leftmost point in the domain|Real
|`domain/xf`|The rightmost point in the domain|Real
|`problem/problem_type`|The type of problem to solve with the QCNN|`"fit"`, `"hagen-poiseuille"`|
|`output/folder_name`| Absolute path of the output folder of the training data| String|

# Singularity image

To avoid problems when running on clusters, `QCNN-Metrics` provides a recipe for building a Singularity image in the `sing` folder, together with a Slurm script and a bash script that takes care of job submission and virtual environment activation on a cluster. Note that the cluster must have Singularity installed for this to work.

## Building the image

In a machine where you have root access (this is mandatory), from inside the `QCNN-Metrics` folder issue

```bash
sudo singularity build qcnn.simg singularity/qcnn.recipe
```

After the image is built, copy it over to the cluster where you will be running the code.

## Submitting the job

TODO: This section needs more detail

```bash
sbatch --job-name=fit_0 job.slurm /home/lucas.t/qcnn.simg /home/lucas.t/fit.toml /home/lucas.t/QCNN-Metrics
```