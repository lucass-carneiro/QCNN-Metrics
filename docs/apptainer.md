# Containers

When running in an HPC cluster, users often don't have access to administrative rights. This can cause issues if the system-wide version of python is too old or if the cluster does not have `pip` enabled. To avoid such problems, `QCNN-Metrics` provides a recipe for building an Apptainer (formerly Singularity) image in the `sing` folder, together with a Slurm script and a bash script that takes care of job submission and virtual environment activation on a cluster. Note that the cluster must have Apptainer installed for this to work.

# Building the image

In a machine where you have root access (this is mandatory), from inside the `QCNN-Metrics` folder issue

```bash
sudo apptainer build qcnn.simg singularity/qcnn.recipe
```

After the image is built, copy it over to the cluster where you will be running the code.

# Submitting the job

TODO: This section needs more detail

```bash
sbatch --job-name=fit_0 job.slurm /home/lucas.t/qcnn.simg /home/lucas.t/fit.toml /home/lucas.t/QCNN-Metrics
```