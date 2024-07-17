# Parameter file description

The operations of `qcnn.py` are controlled via YAML parameter files. Examples of such files can be found in the `parfiles` folder of `QCNN-Metrics`. In this page, you will find a complete description of all available parameters. Additionally, the `qcnn.json` file inside the `parfiles` folder provides a schema file that can be used in text editors for adding in their writing. In `VSCode`, for instance, they can be used with Red Hat's `YAML Language server` plugin.

## `computer` options

These options describe the simulated quantum computer that shall be used.

* `num_qubits`: The number of `qubits` to use in the computer. Must be an integer larger or equal to 1.

## `circuit` options

These options describe the types of quantum circuits that can be trained and their respective options.

* `ansatz`: The type of ansatz circuit to use. Can be one of the following:
    1. `"sel"`: A [`StronglyEntanglingLayer`](https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html) circuit.
    2. `"conv"`: A convolutional circuit without pooling layers. These circuits act on pairs of quibits and "wrap around", that is, the last qubit in the computer is paired with the first.
* `num_layers`: This parameter is only required if `ansatz` is `"sel"`. It configures the number of entangling layers to use.
* `conv_layer`: The name of the convolutional layer to use. The name of the layer *must* be one of the classes in the `conv_layers.py` file. This string is interpreted directly in the code.

## `dataset` options

These options control the input dataset.

* `dataset_size`: The number of points in the discrete domain.
* `batch_size`: The size of training batches. A batch is a subset of the original input data that can be used for training instead of the whole data set. If set to 0, batching is disabled.

## `training` options

These options control the optimizer algorithm used for training the circuit.

* `optimizer`: The underlying optimizer library to use. Can be one of the following.
    1. `"numpy"`: Uses the bundled Pennylane Numpy Adam optimizer.
    2. `"torch"`: Uses the adaptive L-BGFS optimizer from the PyTorch library.
* `use_cuda`: Uses CUDA tensors for training. This is only available if `optimizer` is `"torch"`.
* `max_iters`: The maximum number of optimizer steps to take.
* `abstol`: The value of the error function below which the optimizer will stop iterating.
* `step_size`: The initial step size of the optimizer.

## `domain` options

The 1D domain where functions will be fitted or solved.

* `x0`: Left boundary of the domain.
* `xf`: Right boundary of the domain.

## `problem` options

These options control the problem type to solve and problem specific parameters.

* `type`: The type of problem to solve. Must be one of the following:
    1. `"fit"`: Fit a target function.
    2. `"hagen-poiseuille"`: Solve the tubular Hagen-Poiseuille equation.
    3. `"plane-hagen-poiseuille"`: Solve the Hagen-Poiseuille equation between two infinite planes.

## `output` options

These options control the data output of the training.

* `folder_name`: The name of the folder where to save the training and plot data.