# Getting started with QCNN-Metrics

The main interaction point of this package is the `qcnn.py` script, provided in the `src` folder of this repository.

This script takes a single argument, a configuration file, that completely drives the behavior of the program. The next few sections will teach you to use its usage.

# Installing dependencies

In order to use the `qcnn.py` script, dependencies need to be installed. The recommended way to install the required dependencies is by creating a virtual environment and using pip to install dependencies to that virtual environment. All the package's dependencies are specified in the `requirements.txt` file within the repository. To create a suitable virtual environment, issue the following commands from within the `QCNN-Metrics` repository folder:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Issue python commands using the libraries installed in the virtual environment

deactivate # When done with using the environment
```

Note that the virtual environment does not necessarily need to be named `venv` or located inside the `QCNN-Metrics` folder. Just be sure to point to the correct location of the `requirements.txt` file when creating the environment and to activate the correct environment prior to interacting with the code.

# Usage

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