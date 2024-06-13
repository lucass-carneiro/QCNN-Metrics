# Intro

This is a python package for training quantum convolutional neural networks.

# Singularity image

```
sudo singularity build qcnn.simg singularity/qcnn.recipe
sbatch --job-name=fit_0 QCNN-Metrics/sing/job.slurm /home/lucas.t/QCNN-Metrics/parfiles/fit/fit_0.toml
```