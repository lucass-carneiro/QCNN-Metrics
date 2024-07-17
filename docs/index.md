<h1 style="text-align: center;">
QCNN Toolkit
</h1>

---

This is a python package for training quantum convolutional neural networks. It uses `Pennylane` for building quantum circuits and can be trained with `Numpy` or `Torch` backends on CPUs or GPUs. The code also supports checkpoints, recovery and streaming IO via the `ADIOS2` library.

This program trains quantum circuits to solve the following problems:

1. 1D Function fitting
2. ODE solving